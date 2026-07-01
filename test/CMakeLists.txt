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
