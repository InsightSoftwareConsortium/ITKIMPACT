# Test data

Small, reproducible inputs are committed directly and used by the self-contained tests:

- `ImpactToyModel.pt`, `ImpactToyModelDown.pt` — tiny deterministic TorchScript models
  (regenerate with `GenerateImpactTestModels.py`).

Every C++ unit test and GoogleTest synthesizes its images at run time, so no heavy binary
is required to build and run the default suite.

## Optional real-CT regression (out-of-git, ExternalData)

`ImpactConvexAdam.RealLungCTCoarseInitThenAdamRefines` validates the coarse+fine pipeline on
a real lung-CT pair. Those inputs are large, so they are **not** committed:

- `3DCT_lung_baseline.mha`, `3DCT_lung_followup.mha` — inhale/exhale lung CT pair
- `M258_2_Layers.pt` — a real IMPACT feature extractor

They are kept out of git (see `../../.gitignore`); the blobs are hosted on
https://data.kitware.com and fetched on demand via ITK ExternalData, driven by the committed
`*.sha512` content links next to this file. `test/CMakeLists.txt` expands them with `DATA{...}`
and an `ExternalData_add_target`, so the gtest runs in CI; if the download is unavailable the
test **skips** rather than fails. A local copy dropped in this directory is used as-is.

To refresh a file: replace it, regenerate its link (`sha512sum <file> | awk '{print $1}' >
<file>.sha512`), and upload the new blob to data.kitware.com (addressed by SHA512).
