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

#include <vector>

namespace nvbench
{

template <typename Strategy>
struct axis_space_iterator final : public axis_space_iterator_base
{
  using strategy_type = Strategy;

private:
  [[nodiscard]] std::unique_ptr<axis_space_iterator_base> do_clone() const override
  {
    return std::make_unique<axis_space_iterator<strategy_type>>(*this);
  }

  void do_compute_linear_size() override
  {
    m_linear_size = m_strategy.compute_linear_size(const_cast<const indices_type &>(m_axis_sizes));
  }

  void do_update_value_indices() override
  {
    m_strategy.update_value_indices(m_axis_value_indices,
                                    const_cast<const indices_type &>(m_axis_sizes),
                                    const_cast<const std::size_t &>(m_linear_index),
                                    const_cast<const std::size_t &>(m_linear_size));
  }

  // TODO: Document the strategy interface
  strategy_type m_strategy{};
};

} // namespace nvbench
