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

#include <functional>

namespace nvbench
{

struct axis_space_lambda_iterator final : public nvbench::axis_space_iterator_base
{
  using nvbench::axis_space_iterator_base::indices_type;

  // Compute the linear size of the iteration space based on the provided axis sizes.
  using compute_linear_size_signature = std::size_t(const indices_type &axis_sizes);

  // Update the axis value indices based on the current linear iteration index.
  using update_value_indices_signature = void(indices_type &axis_value_indices,
                                              const indices_type &axis_sizes,
                                              const std::size_t linear_idx,
                                              const std::size_t linear_size);

  // Construct an iterator that iterates over the cartesian product of the
  // associated axes.
  axis_space_lambda_iterator() = default;

  // Construct an iterator that uses the provided functions to describe the iteration space.
  axis_space_lambda_iterator(std::function<compute_linear_size_signature> compute_linear_size,
                             std::function<update_value_indices_signature> update_value_indices)
      : m_compute_linear_size{std::move(compute_linear_size)}
      , m_update_value_indices{std::move(update_value_indices)}
  {}

  std::size_t do_compute_linear_size() const override
  {
    return m_compute_linear_size(m_axis_sizes);
  }

  void do_update_value_indices() override
  {
    m_update_value_indices(m_axis_value_indices, m_axis_sizes, m_linear_index, m_linear_size);
  }

private:
  indices_type m_sizes{};
  std::size_t m_idx{};
  std::size_t m_size{};

  // Default implementation generates a cartesian product of the axes.
  std::function<compute_linear_size_signature> m_compute_linear_size =
    [](const indices_type &axes_sizes) -> std::size_t {
    std::size_t iteration_size = 1;
    for (const auto size : axes_sizes)
    {
      iteration_size *= size;
    }
    return iteration_size;
  };

  // Default implementation generates a cartesian product of the axes.
  std::function<update_value_indices_signature> m_update_value_indices =
    [](indices_type &axes_value_indices,
       const indices_type &axes_sizes,
       const std::size_t idx,
       const std::size_t /*size*/) {
      if (idx == 0)
      { // Initialize:
        std::fill(axes_value_indices.begin(), axes_value_indices.end(), 0);
        return;
      }

      // Advance:
      for (std::size_t i = 0; i < axes_value_indices.size(); ++i)
      {
        ++axes_value_indices[i];
        if (axes_value_indices[i] < axes_sizes[i])
        {
          return;
        }
        else
        {
          axes_value_indices[i] = 0;
        }
      }
    };
};

} // namespace nvbench
