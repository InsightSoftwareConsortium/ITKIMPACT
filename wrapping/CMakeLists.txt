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

itk_wrap_module(Impact)
itk_auto_load_submodules()
itk_end_wrap_module()

# The Python module must link the compiled Impact library, which holds the only
# non-inline symbol of the torch-free public API: the ModelConfiguration constructor
# (src/itkModelConfiguration.cxx). The module's external Impact_LIBRARIES is LibTorch
# only, so the library target is linked explicitly here.
if(TARGET ImpactPython AND TARGET Impact)
  target_link_libraries(ImpactPython PUBLIC Impact)
endif()

# LibTorch is not bundled in the wheel (see pyproject [tool.cibuildwheel]): the compiled
# module still records a DT_NEEDED on libtorch, resolved at runtime from the installed
# `torch` package. Point an RPATH from the module (installed as itk/_ImpactPython*.so) at
# that package's lib dir (site-packages/itk -> site-packages/torch/lib) so it loads even
# when `torch` was not imported first. Consumers that already `import torch` (e.g. KonfAI)
# work regardless, since the soname is then already loaded.
if(TARGET ImpactPython)
  if(APPLE)
    set_property(TARGET ImpactPython APPEND PROPERTY INSTALL_RPATH "@loader_path/../torch/lib")
  elseif(UNIX)
    set_property(TARGET ImpactPython APPEND PROPERTY INSTALL_RPATH "$ORIGIN/../torch/lib")
  endif()
endif()
