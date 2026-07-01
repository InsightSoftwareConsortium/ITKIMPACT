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

cmake_minimum_required(VERSION 3.22.1...3.29.0 FATAL_ERROR)
project(Impact)

# LibTorch's headers emit MSVC-only C4267 (size_t -> int narrowing, e.g. in
# torch/csrc/dynamo/compiled_autograd.h) when its templates are instantiated in our
# translation units. The ITK dashboard treats any build warning as a failure, so silence
# this benign third-party warning on MSVC (gcc/clang do not emit it; ITK already suppresses
# the related C4251). Set at the module root so every target -- library, tests, header test
# and wrapping -- inherits it, on both the standalone and in-ITK-tree build paths below.
if(MSVC)
  add_compile_options(/wd4267)
endif()

# Impact is a header-only module; the libraries clients must link (LibTorch) are
# set in itk-module-init.cmake once find_package(Torch) has run.

if(NOT ITK_SOURCE_DIR)
  include(itk-module-init.cmake)
  find_package(ITK REQUIRED)
  list(APPEND CMAKE_MODULE_PATH ${ITK_CMAKE_DIR})
  include(ITKModuleExternal)
else()
  set(ITK_DIR ${CMAKE_BINARY_DIR})
  itk_module_impl()
endif()

itk_module_examples()
