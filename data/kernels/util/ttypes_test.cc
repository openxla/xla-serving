/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Copyright 2026 Google LLC. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "data/kernels/util/ttypes.h"  // from @xla_serving

#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include "runtime/core/host_buffer.h"  // from @xla_serving

namespace xla_serving::data::kernel {
namespace {

TEST(TTypesTest, ScalarAndConstScalar) {
  float value = 42.0f;
  typename TTypes<float>::Scalar scalar_map(&value);
  EXPECT_EQ(scalar_map.rank(), 0);
  EXPECT_EQ(scalar_map(), 42.0f);

  scalar_map() = 100.0f;
  EXPECT_EQ(value, 100.0f);

  typename TTypes<float>::ConstScalar const_scalar_map(&value);
  EXPECT_EQ(const_scalar_map.rank(), 0);
  EXPECT_EQ(const_scalar_map(), 100.0f);
}

TEST(TTypesTest, FlatAndMatrix) {
  std::vector<int> data = {1, 2, 3, 4, 5, 6};
  typename TTypes<int>::Flat flat_map(data.data(), 6);
  EXPECT_EQ(flat_map.rank(), 1);
  EXPECT_EQ(flat_map(0), 1);
  EXPECT_EQ(flat_map(5), 6);

  typename TTypes<int>::Vec vec_map(data.data(), 6);
  EXPECT_EQ(vec_map.rank(), 1);
  EXPECT_EQ(vec_map(0), 1);
  EXPECT_EQ(vec_map(5), 6);

  static_assert(
      std::is_same_v<typename TTypes<int>::Vec, typename TTypes<int>::Flat>);
  static_assert(std::is_same_v<typename TTypes<float>::ConstVec,
                               typename TTypes<float>::ConstFlat>);

  typename TTypes<int, 2>::Matrix matrix_map(data.data(), 2, 3);
  EXPECT_EQ(matrix_map.rank(), 2);
  EXPECT_EQ(matrix_map(0, 0), 1);
  EXPECT_EQ(matrix_map(1, 2), 6);
}

TEST(TTypesTest, CustomIndexType) {
  static_assert(
      std::is_same_v<typename TTypes<float, 1, int32_t>::Tensor::Index,
                     int32_t>);
  static_assert(
      std::is_same_v<typename TTypes<float, 1, int32_t>::Scalar::Index,
                     int32_t>);
  static_assert(
      std::is_same_v<typename TTypes<float, 1, int32_t>::ConstScalar::Index,
                     int32_t>);
  static_assert(
      std::is_same_v<typename TTypes<float, 2>::Tensor32Bit::Index, int32_t>);
  static_assert(
      std::is_same_v<typename TTypes<float>::Flat32Bit::Index, int32_t>);
}

TEST(TTypesTest, ConstTensorEquivalence) {
  static_assert(std::is_same_v<typename TTypes<float, 4>::ConstTensor,
                               typename TTypes<const float, 4>::Tensor>);
  static_assert(std::is_same_v<typename TTypes<float, 4>::ConstTensor32Bit,
                               typename TTypes<const float, 4>::Tensor32Bit>);
  static_assert(std::is_same_v<typename TTypes<float>::ConstFlat,
                               typename TTypes<const float>::Flat>);
  static_assert(std::is_same_v<typename TTypes<float>::ConstVec,
                               typename TTypes<const float>::Vec>);
  static_assert(std::is_same_v<typename TTypes<float>::ConstFlat32Bit,
                               typename TTypes<const float>::Flat32Bit>);
  static_assert(std::is_same_v<typename TTypes<float, 2>::ConstMatrix,
                               typename TTypes<const float, 2>::Matrix>);
}

TEST(TTypesTest, PreservesJsvTTypesBase) {
  static_assert(std::is_base_of_v<::xla_serving::TTypes<float, 2, Eigen::DenseIndex>,
                                  TTypes<float, 2, Eigen::DenseIndex>>);
}

TEST(TTypesTest, Index32Type) {
  static_assert(std::is_same_v<Index32, int32_t>);
}

TEST(TTypesTest, SafeFor32BitIndexingDSizes) {
  Eigen::DSizes<int64_t, 2> valid(100, 200);
  EXPECT_TRUE(SafeFor32BitIndexing(valid));

  Eigen::DSizes<int64_t, 2> invalid(
      static_cast<int64_t>(std::numeric_limits<Index32>::max()) + 1, 200);
  EXPECT_FALSE(SafeFor32BitIndexing(invalid));
}

TEST(TTypesTest, SafeFor32BitIndexingArray) {
  Eigen::array<int64_t, 3> valid = {10, 20, 30};
  EXPECT_TRUE(SafeFor32BitIndexing(valid));

  Eigen::array<int64_t, 3> invalid = {
      10, static_cast<int64_t>(std::numeric_limits<Index32>::max()) + 1, 30};
  EXPECT_FALSE(SafeFor32BitIndexing(invalid));
}

TEST(TTypesTest, To32BitConversion) {
  Eigen::DSizes<int64_t, 2> in(10, 20);
  auto out = To32Bit(in);
  EXPECT_EQ(out[0], 10);
  EXPECT_EQ(out[1], 20);

  Eigen::array<int64_t, 2> arr_in = {15, 25};
  auto arr_out = To32Bit(arr_in);
  EXPECT_EQ(arr_out[0], 15);
  EXPECT_EQ(arr_out[1], 25);
}

TEST(TTypesTest, TensorMapTo32Bit) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  Eigen::DSizes<Eigen::DenseIndex, 2> dims(2, 2);
  TTypes<float, 2>::Tensor tensor_map(data.data(), dims);

  EXPECT_TRUE(SafeFor32BitIndexing(tensor_map));
  auto tensor_32bit = To32Bit(tensor_map);
  EXPECT_EQ(tensor_32bit.dimensions()[0], 2);
  EXPECT_EQ(tensor_32bit.dimensions()[1], 2);
  EXPECT_EQ(tensor_32bit(0, 0), 1.0f);
  EXPECT_EQ(tensor_32bit(1, 1), 4.0f);
}

TEST(TTypesTest, MaybeWith32BitIndexingGpuDeviceSafe) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  Eigen::DSizes<Eigen::DenseIndex, 2> dims(2, 2);
  TTypes<float, 2>::Tensor tensor_map(data.data(), dims);

  bool called_with_32bit = false;
  MaybeWith32BitIndexing<Eigen::GpuDevice>(
      [&](auto tensor) {
        if constexpr (std::is_same_v<typename decltype(tensor)::Index,
                                     Index32>) {
          called_with_32bit = true;
          EXPECT_EQ(tensor.dimensions()[0], 2);
          EXPECT_EQ(tensor.dimensions()[1], 2);
          EXPECT_EQ(tensor(0, 0), 1.0f);
          EXPECT_EQ(tensor(1, 1), 4.0f);
        }
      },
      tensor_map);

  EXPECT_TRUE(called_with_32bit);
}

TEST(TTypesTest, MaybeWith32BitIndexingGpuDeviceNotSafe) {
  Eigen::DSizes<int64_t, 2> dims(
      static_cast<int64_t>(std::numeric_limits<Index32>::max()) + 1, 2);
  bool called_with_64bit = false;
  MaybeWith32BitIndexing<Eigen::GpuDevice>(
      [&](auto d) {
        if constexpr (std::is_same_v<std::decay_t<decltype(d[0])>, int64_t>) {
          called_with_64bit = true;
          EXPECT_EQ(d[1], 2);
        }
      },
      dims);

  EXPECT_TRUE(called_with_64bit);
}

TEST(TTypesTest, MaybeWith32BitIndexingCpuDevice) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  Eigen::DSizes<Eigen::DenseIndex, 2> dims(2, 2);
  TTypes<float, 2>::Tensor tensor_map(data.data(), dims);

  bool called_with_64bit = false;
  MaybeWith32BitIndexing<Eigen::DefaultDevice>(
      [&](auto tensor) {
        if constexpr (std::is_same_v<typename decltype(tensor)::Index,
                                     Eigen::DenseIndex>) {
          called_with_64bit = true;
          EXPECT_EQ(tensor(0, 0), 1.0f);
        }
      },
      tensor_map);

  EXPECT_TRUE(called_with_64bit);
}

}  // namespace
}  // namespace xla_serving::data::kernel
