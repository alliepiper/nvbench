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

#include <nvbench/axis_space_iterator_base.cuh>

#include <nvbench/axes_metadata.cuh>

namespace nvbench
{

axis_space_iterator_base::~axis_space_iterator_base() = default;

void axis_space_iterator_base::initialize(const axes_metadata &axes)
{
  m_axis_value_indices.clear();
  m_axis_value_indices.resize(m_axis_indices.size(), 0);

  m_axis_sizes.clear();
  m_axis_sizes.reserve(m_axis_indices.size());
  for (const auto &axis_idx : m_axis_indices)
  {
    const auto &axis = *axes.get_axes()[axis_idx];
    m_axis_sizes.push_back(axis.get_size());
  }

  m_linear_index = 0;
  m_linear_size = this->do_compute_linear_size(m_axis_sizes);

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

}
