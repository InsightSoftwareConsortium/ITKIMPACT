#!/usr/bin/env bash
# Compare the speed of two revisions of ITKIMPACT on this machine, busy or not.
#
#   benchmarks/ab.sh BUILD_DIR REF [REV] [ROUNDS]
#
# A time alone means little on a machine that also runs other jobs; two revisions run side by side
# under the same load do. Each revision (REV defaults to HEAD) is exported to its own tree under
# ${IMPACT_AB_CACHE:-~/.cache/itkimpact-ab}, gets benchmarks/ImpactPerfBench.cxx from this
# checkout appended to its GTest driver (the same measuring code for both), and is built like
# BUILD_DIR, a configured ITKIMPACT build: same ITK, LibTorch, compilers and link flags. The test
# data is fetched once into the cache, for every revision. A revision is built once per version of
# the bench and kept.
#
# The two drivers then run alternately, ROUNDS times (default 5). Printed per case: the median of
# each revision over the rounds, their ratio, and how many rounds REV won. The script exits 1 when,
# on a case, REV is slower by more than ${IMPACT_AB_THRESHOLD:-10} % in the median and in at least
# three rounds out of four; 2 when a case was not measured on both sides, or none at all (missing
# data, a filter that keeps nothing): a comparison that measured nothing is not a pass.
#
# IMPACT_PERF_ONLY keeps the cases whose name (mode-device-loss) contains it: online-cpu,
# online-cuda, static, elastix-cpu (the elastix metric's online path), ncc...
# IMPACT_PERF_LOCK names a file held with flock around every run, for a GPU shared with other jobs.
# The drivers run in this environment: it must let them load LibTorch, as for ctest.
set -euo pipefail

if [ $# -lt 2 ]; then
  sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'
  exit 2
fi
build_dir=$(cd "$1" && pwd)
ref=$2
rev=${3:-HEAD}
rounds=${4:-5}
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
bench="$repo/benchmarks/ImpactPerfBench.cxx"
cache=${IMPACT_AB_CACHE:-$HOME/.cache/itkimpact-ab}
mkdir -p "$cache"

# The configuration of BUILD_DIR that a revision's build has to share.
config=()
for key in ITK_DIR Torch_DIR CMAKE_BUILD_TYPE CMAKE_C_COMPILER CMAKE_CXX_COMPILER CMAKE_CUDA_COMPILER \
  CMAKE_CUDA_ARCHITECTURES CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS; do
  line=$(grep -m1 "^$key:" "$build_dir/CMakeCache.txt" || true)
  if [ -n "$line" ]; then
    config+=("-D${line%%=*}=${line#*=}")
  fi
done

# The GTest driver of a revision, built if it is not yet; its path on stdout.
driver() {
  local sha key source build
  sha=$(git -C "$repo" rev-parse --verify "$1^{commit}")
  key="$sha-$(git hash-object "$bench" | cut -c1-12)" # a changed bench is a different build
  source="$cache/$key/source"
  build="$cache/$key/build"
  if [ ! -x "$build/bin/ImpactGTestDriver" ]; then
    echo "building $1 ($sha)" >&2
    rm -rf "$source"
    mkdir -p "$source"
    git -C "$repo" archive "$sha" | tar -x -C "$source"
    cat "$bench" >>"$source/test/itkImpactBackendGTest.cxx"
    # The outputs stay in the build tree: an external module writes into ITK_DIR by default.
    cmake -S "$source" -B "$build" "${config[@]}" -DBUILD_TESTING=ON \
      -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$build/bin" -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$build/lib" \
      -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY="$build/lib" \
      -DExternalData_OBJECT_STORES="$cache/objects" >&2
    cmake --build "$build" --target ImpactGTestDriver -j "$(nproc)" >&2
  fi
  echo "$build/bin/ImpactGTestDriver"
}

ref_driver=$(driver "$ref")
rev_driver=$(driver "$rev")
lock=()
if [ -n "${IMPACT_PERF_LOCK:-}" ]; then
  lock=(flock "$IMPACT_PERF_LOCK")
fi
results=$(mktemp)
trap 'rm -f "$results"' EXIT
for round in $(seq "$rounds"); do
  for side in ref rev; do
    exe=$ref_driver
    if [ $side = rev ]; then
      exe=$rev_driver
    fi
    "${lock[@]}" "$exe" --gtest_also_run_disabled_tests --gtest_filter='ImpactPerf.*' |
      awk -v side=$side -v round="$round" '$1 == "IMPACT_PERF" { print side, round, $2, $3 }' >>"$results"
  done
  echo "round $round of $rounds, load $(cut -d' ' -f1 /proc/loadavg)" >&2
done

python3 - "$results" "$ref" "$rev" "${IMPACT_AB_THRESHOLD:-10}" <<'EOF'
import collections, statistics, sys

path, ref, rev, threshold = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]) / 100
times = collections.defaultdict(dict)  # (case, side) -> {round: ms}
for line in open(path):
    side, round_, case, ms = line.split()
    times[(case, side)][round_] = float(ms)
cases = list(dict.fromkeys(case for case, _ in times))
if not cases:
    print("no case measured: the data is missing or IMPACT_PERF_ONLY keeps none", file=sys.stderr)
    sys.exit(2)
print(f"| case | {ref} (ms) | {rev} (ms) | ratio | rounds {rev} won |")
print("|---|---|---|---|---|")
regressed, missing = [], []
for case in cases:
    a, b = times[(case, "ref")], times[(case, "rev")]
    paired = [r for r in a if r in b]
    if not paired:
        print(f"| {case} | | | | missing |")
        missing.append(case)
        continue
    ratio = statistics.median(b[r] for r in paired) / statistics.median(a[r] for r in paired)
    won = sum(b[r] < a[r] for r in paired)
    print(f"| {case} | {statistics.median(a[r] for r in paired):.1f} | {statistics.median(b[r] for r in paired):.1f}"
          f" | {ratio:.2f} | {won}/{len(paired)} |")
    if ratio > 1 + threshold and (len(paired) - won) >= 0.75 * len(paired):
        regressed.append(case)
if regressed:
    print(f"\n{rev} is slower than {ref} on: {', '.join(regressed)}")
if missing:
    print(f"\nnot measured on both sides: {', '.join(missing)}", file=sys.stderr)
sys.exit(1 if regressed else 2 if missing else 0)
EOF
