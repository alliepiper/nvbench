/*
 *  Copyright 2021-2025 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 with the LLVM exception
 *  (the "License"); you may not use this file except in compliance with
 *  the License.
 *
 *  You may obtain a copy of the License at
 *
 *      http://llvm.org/foundation/relicensing/LICENSE.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <nvbench/axis_base.cuh>
#include <nvbench/type_axis.cuh>

#include <string>

namespace nvbench::detail
{

// Helper for state_generator to track current value and axis information used while bonstructing
// state configs.
struct axis_value_descriptor
{
  explicit axis_value_descriptor(const axis_base &axis)
      : axis_name(axis.get_name())
      , axis_type(axis.get_type())
      , axis_size(axis.get_size())
      , value_index(0)
  {}

  std::string axis_name;
  nvbench::axis_type axis_type;
  std::size_t axis_size;
  std::size_t value_index;
};

} // namespace nvbench::detail
