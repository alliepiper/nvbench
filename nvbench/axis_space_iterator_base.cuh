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

#include <vector>

namespace nvbench
{

class axes_metadata;

struct axis_space_iterator_base
{
  using indices_type = std::vector<std::size_t>;

  axis_space_iterator_base() = default;

  virtual ~axis_space_iterator_base();

  // Which axes this iterator is associated with, specified by their
  // indices in the axes metadata.
  [[nodiscard]] const indices_type &get_axis_indices() const { return m_axis_indices; }
  void set_axis_indices(indices_type axis_indices) { m_axis_indices = std::move(axis_indices); }

  // Initialize the iteration state based on the current axes metadata, using
  // the previously specified axis_indices.
  void initialize(const nvbench::axes_metadata &axes);

  // Advance the iteration state. Returns true if the iterator "rolled over", eg.
  // was exhausted and has reset to the beginning.
  [[nodiscard]] bool advance();

  // The index of the current value in each associated axis.
  [[nodiscard]] const indices_type &get_axis_value_indices() const { return m_axis_value_indices; }

  [[nodiscard]] std::size_t get_linear_index() const { return m_linear_index; }
  [[nodiscard]] std::size_t get_linear_size() const { return m_linear_size; }

protected:
  // The indices of the associated axes in the axes metadata.
  indices_type m_axis_indices{};

  // The indices of the current values in each associated axis.
  indices_type m_axis_value_indices{};

  // The sizes of each associated axis.
  indices_type m_axis_sizes{};

  // The current linear index in the iteration space.
  std::size_t m_linear_index{};

  // The linear size of the iteration space.
  std::size_t m_linear_size{};

  // Compute the linear size of the iteration space based on m_axis_sizes.
  virtual std::size_t do_compute_linear_size() const = 0;

  // Update the value indices based on the current linear_index and/or previous value_indices.
  // Will be called exactly once per iteration.
  // The linear_index may reset to 0 between calls, but will otherwise always be one more than
  // the previous call.
  // The value_indices are preserved between calls.
  virtual void do_update_value_indices() = 0;
};

} // namespace nvbench
