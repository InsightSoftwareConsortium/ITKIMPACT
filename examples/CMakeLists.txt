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
project(ImpactExamples)

set(ExampleSpecificComponents
  Impact
  )

if(NOT ITK_SOURCE_DIR)
  find_package(ITK REQUIRED COMPONENTS ITKImageIO ITKTransformIO ${ExampleSpecificComponents})
else()
  # When being built as part of ITK, ITKImageIO and ITKTransformIO
  # lists of modules are not yet ready, causing a configure error
  find_package(ITK REQUIRED COMPONENTS ${ExampleSpecificComponents})
endif()
include(${ITK_USE_FILE})

add_executable(ImpactMetricExample ImpactMetricExample.cxx )
target_link_libraries(ImpactMetricExample ${ITK_LIBRARIES})
