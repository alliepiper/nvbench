/*
 *  Copyright 2022 NVIDIA Corporation
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

#pragma once

#include <nvbench/axis_space_iterator_base.cuh>
#include <nvbench/detail/axis_value_descriptor.cuh>

namespace nvbench
{
struct axes_metadata;

/*!
 * Base class for all axis iteration spaces.
 *
 * If we consider an axis to be a container of values, iteration_spaces
 * would be how we can create iterators over that container.
 *
 * With that in mind we get the following mapping:
 * * linear_axis_space is equivalent to a forward iterator.
 *
 * * zip_axis_space is equivalent to a zip iterator.
 *
 * * user_axis_space is equivalent to a transform iterator.
 *
 * The `nvbench::axes_metadata` stores all axes in a std::vector. To represent
 * which axes each space is 'over' we store those indices. We don't store
 * the pointers or names for the following reasons:
 *
 * * The names of an axis can change after being added. The `nvbench::axes_metadata`
 * is not aware of the name change, and can't inform this class of it.
 *
 * * The `nvbench::axes_metadata` can be deep copied, which would invalidate
 * any pointers held by this class. By holding onto the index we remove the need
 * to do any form of fixup on deep copies of `nvbench::axes_metadata`.
 */
struct axis_space_base
{
  using axis_value_descriptors = std::vector<detail::axis_value_descriptor>;

  /*!
   * Construct a new derived iteration_space
   *
   * @param[input_axis_indices] Index of each associated axis in axes_metadata.
   */
  axis_space_base(std::vector<std::size_t> input_axis_indices)
      : m_axis_indices(std::move(input_axis_indices))
  {}

  virtual ~axis_space_base();

  [[nodiscard]] std::unique_ptr<axis_space_base> clone() const { return this->do_clone(); }

  /*!
   * Create and return an iterator over the axes in the iteration space.
   */
  [[nodiscard]] std::unique_ptr<axis_space_iterator_base>
  get_iterator(const axes_metadata &axes) const
  {
    return this->do_get_iterator(axes);
  }

  /*!
   * Returns the number of active and inactive elements the iterator will have
   * when executed over @a axes
   *
   * Note:
   *  Type Axis support inactive elements
   */
  [[nodiscard]] std::size_t get_size(const axes_metadata &axes) const
  {
    return this->do_get_size(axes);
  }

protected:
  std::vector<std::size_t> m_axis_indices;

  virtual std::unique_ptr<axis_space_base> do_clone() const = 0;
  virtual std::unique_ptr<axis_space_iterator_base>
  do_get_iterator(const axes_metadata &axes) const                 = 0;
  virtual std::size_t do_get_size(const axes_metadata &axes) const = 0;
};

} // namespace nvbench
