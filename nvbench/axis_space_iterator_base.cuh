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

#include <cstdint>
#include <memory>
#include <vector>

namespace nvbench
{
struct axes_metadata;

struct axis_space_iterator_base
{
  using indices_type = std::vector<std::size_t>;

  axis_space_iterator_base() = default;
  virtual ~axis_space_iterator_base();

  [[nodiscard]] std::unique_ptr<axis_space_iterator_base> clone() const;

  // Initialize the iteration state based on the current axes metadata and axis_indices.
  void initialize(const nvbench::axes_metadata &axes, const indices_type &axis_indices);

  // Advance the iteration state. Returns true if the iterator "rolled over", eg.
  // was exhausted and has reset to the beginning.
  [[nodiscard]] bool advance();

  // The index of the current value in each associated axis.
  [[nodiscard]] const indices_type &get_axis_value_indices() const { return m_axis_value_indices; }

  [[nodiscard]] std::size_t get_linear_index() const { return m_linear_index; }
  [[nodiscard]] std::size_t get_linear_size() const { return m_linear_size; }

protected:
  virtual std::unique_ptr<axis_space_iterator_base> do_clone() const = 0;

  // Compute the linear size of the iteration space based on m_axis_sizes.
  // Only called once during initialization.
  virtual void do_compute_linear_size() = 0;

  // Update the value indices based on the current linear_index and/or previous value_indices.
  // Will be called over the interval [0, m_linear_size) repeatedly. It will be called exactly
  // once for each linear iteration during each iteration. The value_indices are preserved between
  // calls.
  virtual void do_update_value_indices() = 0;

  // The indices of the current values in each associated axis.
  indices_type m_axis_value_indices{};

  // The sizes of each associated axis.
  indices_type m_axis_sizes{};

  // The current linear index in the iteration space.
  std::size_t m_linear_index{};

  // The linear size of the iteration space.
  std::size_t m_linear_size{};
};

} // namespace nvbench
