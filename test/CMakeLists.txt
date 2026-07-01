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

itk_module_test()

set(Impact-Test_LIBRARIES
  ${Impact-Test_LIBRARIES}
  ${ITK_LIBRARIES}
  Impact
)

set(ImpactTests
  itkImageToTensorFilterTest.cxx
  itkTensorToImageFilterTest.cxx
  itkImageToFeaturesMapTest.cxx
  itkInterpolateVectorImageFunctionTest.cxx)

CreateTestDriver(Impact "${Impact-Test_LIBRARIES}" "${ImpactTests}")

# These tests are self-contained: each synthesizes its own small input at run time.
# itkImageToFeaturesMapTest runs the committed tiny toy model (Data/ImpactToyModel.pt).
itk_add_test(NAME itkImageToTensorFilterTest
  COMMAND ImpactTestDriver itkImageToTensorFilterTest)

itk_add_test(NAME itkTensorToImageFilterTest
  COMMAND ImpactTestDriver itkTensorToImageFilterTest)

itk_add_test(NAME itkImageToFeaturesMapTest
  COMMAND ImpactTestDriver itkImageToFeaturesMapTest
  ${CMAKE_CURRENT_SOURCE_DIR}/Data/ImpactToyModel.pt)

itk_add_test(NAME itkInterpolateVectorImageFunctionTest
  COMMAND ImpactTestDriver itkInterpolateVectorImageFunctionTest)

# GoogleTest-based unit tests for the IMPACT backend. Self-contained: they use
# synthetic images plus the tiny committed Data/ImpactToyModel.pt, so they do
# not depend on ExternalData.
#
# The driver carries its own main() (see itkImpactBackendGTest.cxx) instead of
# GTest::Main, because the shared ITK NrrdIO library exports a main() from its
# sampleIO demo; a direct object's main() takes precedence, whereas the gtest
# main archive would be skipped once the shared one resolves the symbol. We
# therefore link the ITK-provided GTest::GTest (the library only, no main).
add_executable(ImpactGTestDriver itkImpactBackendGTest.cxx)
target_link_libraries(ImpactGTestDriver ${Impact-Test_LIBRARIES} Impact GTest::GTest)
# When ITK is built with ITK_WRAP_PYTHON=ON, ITKCommon embeds itk::PyCommand (Python C
# API), so executables linking ITK must also link libpython (the embed library).
if(ITK_WRAP_PYTHON)
  find_package(Python3 COMPONENTS Development.Embed QUIET)
  set(_impact_pylib "")
  if(TARGET Python3::Python)
    set(_impact_pylib Python3::Python)
  elseif(IMPACT_PYTHON_EMBED_LIBRARY)
    set(_impact_pylib ${IMPACT_PYTHON_EMBED_LIBRARY})
  endif()
  # --no-as-needed: the Python C-API symbols are referenced by libITKCommon (a shared
  # object), not by this executable's own objects, so the default --as-needed would
  # drop libpython and leave them unresolved.
  if(_impact_pylib)
    target_link_libraries(ImpactGTestDriver -Wl,--no-as-needed ${_impact_pylib} -Wl,--as-needed)
  endif()
endif()
# The ImpactConvexAdam.RealLungCT* gtest runs on a real lung-CT pair (and the M258 model)
# kept out of git. Fetch them via ITK ExternalData from the committed .sha512 content links
# (blobs hosted on data.kitware.com); the gtest skips gracefully if they are unavailable.
include(ExternalData)
ExternalData_expand_arguments(ImpactData _impactRealCTBaseline DATA{Data/3DCT_lung_baseline.mha})
ExternalData_expand_arguments(ImpactData _impactRealCTFollowup DATA{Data/3DCT_lung_followup.mha})
ExternalData_expand_arguments(ImpactData _impactRealCTModel DATA{Data/M258_2_Layers.pt})
get_filename_component(_impactExternalDataDir "${_impactRealCTBaseline}" DIRECTORY)
ExternalData_add_target(ImpactData)
add_dependencies(ImpactGTestDriver ImpactData)

target_compile_definitions(ImpactGTestDriver PRIVATE
  IMPACT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Data"
  IMPACT_EXTERNAL_DATA_DIR="${_impactExternalDataDir}")
itk_module_target_label(ImpactGTestDriver)

# Windows has no RPATH: the test drivers link LibTorch, but the loader cannot find the torch
# DLLs at run time (STATUS_DLL_NOT_FOUND, 0xC0000135) -- which breaks both ctest and the
# gtest_discover_tests build-time run below. Copy the LibTorch runtime DLLs next to the drivers
# so Windows resolves them from the executable's own directory (Linux/macOS use RPATH instead).
# This must precede gtest_discover_tests so the DLLs exist before its POST_BUILD discovery run.
if(WIN32)
  if(NOT DEFINED TORCH_INSTALL_PREFIX)
    find_package(Torch REQUIRED)
  endif()
  file(GLOB _impact_torch_dlls "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  foreach(_impact_driver ImpactTestDriver ImpactGTestDriver)
    if(TARGET ${_impact_driver} AND _impact_torch_dlls)
      add_custom_command(TARGET ${_impact_driver} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_impact_torch_dlls} "$<TARGET_FILE_DIR:${_impact_driver}>"
        COMMAND_EXPAND_LISTS
        COMMENT "Copying LibTorch DLLs next to ${_impact_driver} (Windows runtime dependency)")
    endif()
  endforeach()
endif()

include(GoogleTest)
gtest_discover_tests(ImpactGTestDriver DISCOVERY_TIMEOUT 120)
