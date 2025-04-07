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

#include <nvbench/benchmark.cuh>
#include <nvbench/benchmark_manager.cuh>
#include <nvbench/callable.cuh>
#include <nvbench/exec_tag.cuh>
#include <nvbench/type_list.cuh>

#include <memory>

#include <type_traits>

namespace nvbench::create::detail
{

template <typename ExecTagT  = decltype(nvbench::exec_tag::none),
          typename TypeListT = nvbench::type_list<>>
struct static_params
{
  using exec_tag_t  = ExecTagT;
  using type_list_t = TypeListT;
};

[[nodiscard]] constexpr auto make_static_params() { return static_params<>{}; }

template <typename T1>
[[nodiscard]] constexpr auto make_static_params(T1)
{
  if constexpr (nvbench::is_exec_tag_v<T1>)
  {
    return static_params<T1>{};
  }
  else if constexpr (nvbench::is_type_list_v<T1>)
  {
    return static_params<decltype(nvbench::exec_tag::none), T1>{};
  }
  else
  {
    static_assert(nvbench::is_exec_tag_v<T1> || nvbench::is_type_list_v<T1>,
                  "NVBENCH_CREATE(KernelGenerator, __?__) extra arg must be an nvbench::exec_tag "
                  "or nvbench::type_list.");
  }
}

template <typename T1, typename T2>
[[nodiscard]] constexpr auto make_static_params(T1, T2)
{
  if constexpr (nvbench::is_exec_tag_v<T1>)
  {
    static_assert(nvbench::is_type_list_v<T2>,
                  "NVBENCH_CREATE(KernelGenerator, ExecTag, __?__) third argument must be an "
                  "nvbench::type_list.");
    return static_params<T1, T2>{};
  }
  else if constexpr (nvbench::is_type_list_v<T1>)
  {
    static_assert(nvbench::is_exec_tag_v<T2>,
                  "NVBENCH_CREATE(KernelGenerator, TypeAxes, __?__) third argument must be an "
                  "nvbench::exec_tag.");
    return static_params<T2, T1>{};
  }
  else
  {
    static_assert((nvbench::is_exec_tag_v<T1> || nvbench::is_type_list_v<T1>) &&
                    (nvbench::is_exec_tag_v<T2> || nvbench::is_type_list_v<T2>),
                  "NVBENCH_CREATE(KernelGenerator, __?__, __?__) extra args must be an "
                  "nvbench::exec_tag amd nvbench::type_list.");
  }
}

template <typename KernelGenerator, typename StaticParams = static_params<>>
[[nodiscard]] std::unique_ptr<benchmark_base> create_benchmark(KernelGenerator, StaticParams)
{
  using exec_tag_t  = typename StaticParams::exec_tag_t;
  using type_list_t = typename StaticParams::type_list_t;
  return std::make_unique<benchmark<KernelGenerator, exec_tag_t, type_list_t>>();
}

} // namespace nvbench::create::detail

#define NVBENCH_CREATE_TYPE_AXES(...)                                                              \
  nvbench::type_list<__VA_ARGS__> {}

#define NVBENCH_CREATE(KernelGenerator, ...)                                                       \
  NVBENCH_DEFINE_UNIQUE_CALLABLE(KernelGenerator);                                                 \
  nvbench::benchmark_base &NVBENCH_UNIQUE_IDENTIFIER(obj_##KernelGenerator) =                      \
    nvbench::benchmark_manager::get()                                                              \
      .add(nvbench::create::detail::create_benchmark(NVBENCH_UNIQUE_IDENTIFIER(KernelGenerator){}, \
                                                     nvbench::create::detail::make_static_params(  \
                                                       __VA_ARGS__)))                              \
      .set_name(#KernelGenerator)

// TODO deprecate / remove
#define NVBENCH_TYPE_AXES(...) nvbench::type_list<__VA_ARGS__>

// TODO deprecate / remove
#define NVBENCH_BENCH(KernelGenerator) NVBENCH_CREATE(KernelGenerator)

// TODO deprecate / remove
#define NVBENCH_BENCH_TYPES(KernelGenerator, TypeAxes) NVBENCH_CREATE(KernelGenerator, TypeAxes{})
