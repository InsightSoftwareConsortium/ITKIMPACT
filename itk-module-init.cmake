#==========================================================================
#
#   Copyright NumFOCUS
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#          https://www.apache.org/licenses/LICENSE-2.0.txt
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#==========================================================================

# LibTorch is provided by the installed `torch` Python package (the libtorch it bundles),
# not a separately downloaded C++ distribution. One wheel then inherits CPU or CUDA from
# whatever torch the user installed, and we never ship a second multi-GB copy of LibTorch.
# Auto-locate it via `torch.utils.cmake_prefix_path` unless the caller already pointed
# CMAKE_PREFIX_PATH / Torch_DIR at a LibTorch (e.g. a manual C++ distribution).
#
# torch is a hard dependency, so if the `torch` package is missing we pip-install the CPU
# build (best effort) rather than fail outright — this lets the standard ITK CI (which does
# not provision LibTorch) build the module out of the box. Turn it off with
# -DIMPACT_AUTO_INSTALL_TORCH=OFF, or point -DTorch_DIR / -DCMAKE_PREFIX_PATH at a manual
# (e.g. CUDA) LibTorch, which always takes precedence.
option(IMPACT_AUTO_INSTALL_TORCH
  "If the torch Python package is missing, pip-install the CPU build so find_package(Torch) succeeds." ON)

if(NOT Torch_DIR AND NOT DEFINED Torch_ROOT)
  # Prefer the interpreter CMake was configured with (e.g. the cp3xx wheel-build python in
  # the manylinux container) — a bare `python3` on PATH can be an old system Python for which
  # a current torch has no wheel. Fall back to a python3 on PATH only if none was provided.
  if(DEFINED Python3_EXECUTABLE AND EXISTS "${Python3_EXECUTABLE}")
    set(IMPACT_Python_EXECUTABLE "${Python3_EXECUTABLE}")
  elseif(DEFINED PYTHON_EXECUTABLE AND EXISTS "${PYTHON_EXECUTABLE}")
    set(IMPACT_Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
  else()
    find_program(IMPACT_Python_EXECUTABLE NAMES python3 python)
  endif()
  if(IMPACT_Python_EXECUTABLE)
    execute_process(
      COMMAND "${IMPACT_Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
      OUTPUT_VARIABLE IMPACT_Torch_cmake_prefix
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE IMPACT_Torch_query
      ERROR_QUIET)

    if(NOT IMPACT_Torch_query EQUAL 0 AND IMPACT_AUTO_INSTALL_TORCH)
      message(STATUS "IMPACT: the torch Python package was not found; installing the CPU build via pip ...")
      execute_process(
        COMMAND "${IMPACT_Python_EXECUTABLE}" -m pip install "torch==2.12.*"
                --index-url https://download.pytorch.org/whl/cpu
        RESULT_VARIABLE IMPACT_Torch_pip)
      if(NOT IMPACT_Torch_pip EQUAL 0)
        # Some distributions ship an externally-managed Python (PEP 668); retry allowing it.
        execute_process(
          COMMAND "${IMPACT_Python_EXECUTABLE}" -m pip install --break-system-packages "torch==2.12.*"
                  --index-url https://download.pytorch.org/whl/cpu
          RESULT_VARIABLE IMPACT_Torch_pip)
      endif()
      if(IMPACT_Torch_pip EQUAL 0)
        execute_process(
          COMMAND "${IMPACT_Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
          OUTPUT_VARIABLE IMPACT_Torch_cmake_prefix
          OUTPUT_STRIP_TRAILING_WHITESPACE
          RESULT_VARIABLE IMPACT_Torch_query
          ERROR_QUIET)
      endif()
    endif()

    if(IMPACT_Torch_query EQUAL 0 AND IMPACT_Torch_cmake_prefix)
      message(STATUS "IMPACT: using LibTorch from the installed torch package: ${IMPACT_Torch_cmake_prefix}")
      list(APPEND CMAKE_PREFIX_PATH "${IMPACT_Torch_cmake_prefix}")
    endif()
  endif()
endif()

find_package(Torch REQUIRED)

# The module is mostly header-only (templates), plus a small compiled Impact library
# (src/) for the torch-free ModelConfiguration facade. Its public templates still
# instantiate LibTorch in client code, so LibTorch is propagated as an extra library
# clients must link, in addition to the Impact library itself.
set(Impact_LIBRARIES Impact ${TORCH_LIBRARIES})

set(Impact_WRAP_INCLUDE_DIRS
  ${TORCH_INCLUDE_DIRS}
)

set(Impact_SYSTEM_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
set(Impact_SYSTEM_LIBRARY_DIRS ${Torch_LIBRARY_DIRS})

# When this module is consumed by an app, make it re-find Torch so the imported
# torch targets exist downstream.
set(Impact_EXPORT_CODE_INSTALL "
set(Torch_DIR \"${Torch_DIR}\")
find_package(Torch REQUIRED)
")
set(Impact_EXPORT_CODE_BUILD "
if(NOT ITK_BINARY_DIR)
  set(Torch_DIR \"${Torch_DIR}\")
  find_package(Torch REQUIRED)
endif()
")
