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

#include <nvbench/axes_metadata.cuh>
#include <nvbench/axis_space_iterator_base.cuh>

namespace nvbench
{

[[nodiscard]] std::unique_ptr<axis_space_iterator_base> axis_space_iterator_base::clone() const
{
  return std::unique_ptr<axis_space_iterator_base>(this->do_clone());
}

void axis_space_iterator_base::initialize(const axes_metadata &axes,
                                          const indices_type &axis_indices)
{
  m_axis_value_indices.clear();
  m_axis_value_indices.resize(axis_indices.size(), 0);

  m_axis_sizes.clear();
  m_axis_sizes.reserve(axis_indices.size());
  for (const auto &axis_idx : axis_indices)
  {
    const auto &axis = *axes.get_axes()[axis_idx];
    m_axis_sizes.push_back(axis.get_size());
  }

  m_linear_index = 0;

  this->do_compute_linear_size();
  this->do_update_value_indices();
}

bool axis_space_iterator_base::advance()
{
  m_linear_index++;
  if (m_linear_index < m_linear_size)
  {
    this->do_update_value_indices();
    return false;
  }

  m_linear_index = 0;
  this->do_update_value_indices();
  return true; // rolled over
}

} // namespace nvbench
