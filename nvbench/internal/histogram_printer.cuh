/*
 *  Copyright 2021 NVIDIA Corporation
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

#include <nvbench/types.cuh>

#include <fmt/format.h>

#include <cassert>
#include <cmath>
#include <iterator>
#include <numeric>
#include <utility>
#include <vector>

namespace nvbench::internal
{

struct histogram_printer
{
  histogram_printer(nvbench::float64_t min,
                    nvbench::float64_t stride,
                    const std::vector<nvbench::int64_t> &bins)
      : m_min(min)
      , m_stride(stride)
      , m_bins(bins)
      , m_num_bins(bins.size() - 2)
      // 1 / (size of fullest bin):
      , m_scale_factor(1.f / static_cast<nvbench::float32_t>(std::reduce(
                               bins.cbegin(),
                               bins.cend(),
                               nvbench::int64_t{},
                               [](auto a, auto b) { return std::max(a, b); })))
      , m_chart(m_num_bins * m_num_rows, 0)
  {}

  [[nodiscard]] std::string render()
  {
    for (std::size_t bin = 0; bin < m_num_bins; ++bin)
    {
      this->render_bin(bin);
    }

    return this->render_string();
  }

private:
  void render_bin(std::size_t bin)
  {
    // +1 since the first element is the number of bins < m_min
    const auto value        = m_bins[bin + 1];
    const auto scaled_value = static_cast<nvbench::float32_t>(value) *
                              m_scale_factor;

    const auto num_full = static_cast<std::size_t>(scaled_value * m_num_rows);
    const auto partial_frac = std::fmod(scaled_value, 1.f);
    const auto eighths      = static_cast<nvbench::uint8_t>(partial_frac * 8);

    assert(num_full <= m_num_rows);
    for (std::size_t row = 0; row < num_full; ++row)
    { // Filled elements:
      this->write_element(bin, row, 8);
    }

    if (num_full < m_num_rows)
    { // Partial element:
      this->write_element(bin, num_full, eighths);
    }
  }

  void write_element(std::size_t bin, std::size_t row, nvbench::uint8_t eighths)
  {
    // Rows 0 is stored last
    row = m_num_rows - row - 1;

    const auto offset = row * m_num_bins + bin;
    assert(offset < m_chart.size());
    m_chart[offset] = eighths;
  }

  [[nodiscard]] auto get_row(std::size_t row)
  {
    return std::make_pair(m_chart.begin() + row * m_num_bins,
                          m_chart.begin() + (row + 1) * m_num_bins);
  }

  template <typename BufferIterT>
  static auto render_eighth(BufferIterT buffer_iter, nvbench::uint8_t eighth)
  {
    auto do_write = [&](const char *elem) {
      buffer_iter = fmt::format_to(buffer_iter, elem);
    };

    switch (eighth)
    {
      default:
      case 0:
        do_write(" ");
        break;
      case 1:
        do_write(u8"\u2581");
        break;
      case 2:
        do_write(u8"\u2582");
        break;
      case 3:
        do_write(u8"\u2583");
        break;
      case 4:
        do_write(u8"\u2584");
        break;
      case 5:
        do_write(u8"\u2585");
        break;
      case 6:
        do_write(u8"\u2586");
        break;
      case 7:
        do_write(u8"\u2587");
        break;
      case 8:
        do_write(u8"\u2588");
        break;
    }
    return buffer_iter;
  }

  [[nodiscard]] std::string render_string()
  {
    std::vector<char> buffer;
    buffer.reserve(m_chart.size() * 4);
    auto iter = std::back_inserter(buffer);

    for (std::size_t row = 0; row < m_num_rows; ++row)
    {
      iter               = fmt::format_to(iter, "|");
      auto [first, last] = this->get_row(row);
      for (; first != last; ++first)
      {
        iter = this->render_eighth(iter, *first);
      }
      iter = fmt::format_to(iter, "|\n");
    }

    return std::string(buffer.data(), buffer.size());
  }

  nvbench::float64_t m_min;
  nvbench::float64_t m_stride;
  const std::vector<nvbench::int64_t> &m_bins;

  std::size_t m_num_rows{10};
  std::size_t m_num_bins{};
  // 1 / (size of fullest bin):
  nvbench::float32_t m_scale_factor{};

  // encodes each character of the chart as how many eighths of the element
  // should be filled (eg. a value of 4 will be rendered as U+2584 Lower Half
  // Block)
  std::vector<nvbench::uint8_t> m_chart;
};

} // namespace nvbench::internal
