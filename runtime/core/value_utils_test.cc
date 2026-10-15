// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/core/value_utils.h"  // from @xla_serving

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "runtime/core/device_buffer.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/value.h"  // from @xla_serving
#include "runtime/core/value_matcher.h"  // from @xla_serving
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/python/ifrt/test_util.h"  // from @xla
#include "xla/tsl/lib/core/status_test_util.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla

namespace xla_serving {
namespace {

using ::testing::ElementsAre;

// A move-only type to test the move-only type support.
class MoveOnly {
 public:
  explicit MoveOnly(int value) : value_(value) {}
  MoveOnly(const MoveOnly&) = delete;
  MoveOnly& operator=(const MoveOnly&) = delete;
  MoveOnly(MoveOnly&&) = default;
  MoveOnly& operator=(MoveOnly&&) = default;

  int value() const { return value_; }

 private:
  int value_;
};

TEST(PackAsValuesTest, Move) {
  std::vector<int> objects = {1, 2};
  std::vector<Value> values = internal::PackAsValues(std::move(objects));
  EXPECT_THAT(values, ElementsAre(ValueOfType<int>(1), ValueOfType<int>(2)));
}

TEST(PackAsValuesTest, MoveOnlyType) {
  std::vector<MoveOnly> objects;
  int num_objects = 10;
  for (int i = 0; i < num_objects; ++i) {
    objects.push_back(MoveOnly(i));
  }
  std::vector<Value> values = internal::PackAsValues(std::move(objects));
  ASSERT_EQ(values.size(), num_objects);
  for (int i = 0; i < num_objects; ++i) {
    EXPECT_TRUE(values[i].IsType<MoveOnly>());
    EXPECT_EQ(values[i].Get<MoveOnly>().value(), i);
  }
}

TEST(PackAsValuesTest, MapOfMoveOnlyType) {
  absl::flat_hash_map<int, MoveOnly> objects;
  int num_objects = 10;
  for (int i = 0; i < num_objects; ++i) {
    objects.emplace(i, MoveOnly(i));
  }
  absl::flat_hash_map<int, Value> values =
      internal::PackAsValues(std::move(objects));
  ASSERT_EQ(values.size(), num_objects);
  for (int i = 0; i < num_objects; ++i) {
    auto it = values.find(i);
    ASSERT_NE(it, values.end());
    EXPECT_TRUE(it->second.IsType<MoveOnly>());
    EXPECT_EQ(values[i].Get<MoveOnly>().value(), i);
  }
}

TEST(UnpackFromValuesTest, Move) {
  std::vector<Value> values = {Value(1), Value(2)};
  TF_ASSERT_OK_AND_ASSIGN(std::vector<int> objects,
                          internal::UnpackFromValues<int>(std::move(values)));
  EXPECT_THAT(objects, ElementsAre(1, 2));
}

TEST(UnpackFromValuesTest, Copy) {
  const std::vector<Value> values = {Value(1), Value(2)};
  TF_ASSERT_OK_AND_ASSIGN(std::vector<int> objects,
                          internal::UnpackFromValues<int>(values));
  EXPECT_THAT(objects, ElementsAre(1, 2));
}

TEST(UnpackFromValuesTest, MoveOnlyType) {
  std::vector<Value> values;
  int num_objects = 10;
  for (int i = 0; i < num_objects; ++i) {
    values.push_back(Value(MoveOnly(i)));
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<MoveOnly> objects,
      internal::UnpackFromValues<MoveOnly>(std::move(values)));
  for (int i = 0; i < num_objects; ++i) {
    EXPECT_EQ(objects[i].value(), i);
  }
}

TEST(ValueUtilsTest, ReshapeHostBuffer) {
  HostBuffer orig(xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
                  xla::ifrt::Shape({2, 3}));
  orig.flat<float>().setValues({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Value value_orig(std::move(orig));

  TF_ASSERT_OK_AND_ASSIGN(
      Value reshaped_val,
      xla_serving::Reshape(std::move(value_orig), xla::ifrt::Shape({6})));
  ASSERT_TRUE(reshaped_val.IsType<HostBuffer>());
  const HostBuffer& reshaped_hb = reshaped_val.Get<HostBuffer>();
  EXPECT_EQ(reshaped_hb.shape(), xla::ifrt::Shape({6}));
  EXPECT_EQ(reshaped_hb.flat<float>()(0), 1.0f);
  EXPECT_EQ(reshaped_hb.flat<float>()(5), 6.0f);

  // Invalid element count returns error
  HostBuffer invalid_hb1(xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
                         xla::ifrt::Shape({2, 3}));
  EXPECT_FALSE(
      xla_serving::Reshape(Value(std::move(invalid_hb1)), xla::ifrt::Shape({5})).ok());

  // Invalid negative dimension returns error
  HostBuffer invalid_hb2(xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
                         xla::ifrt::Shape({2, 3}));
  EXPECT_FALSE(
      xla_serving::Reshape(Value(std::move(invalid_hb2)), xla::ifrt::Shape({-2, -3}))
          .ok());
}

TEST(ValueUtilsTest, BitcastHostBuffer) {
  HostBuffer orig(xla::ifrt::DType(xla::ifrt::DType::Kind::kU8),
                  xla::ifrt::Shape({4}));
  orig.flat<uint8_t>().setValues({1, 2, 3, 4});
  Value value_orig(std::move(orig));

  xla::ifrt::DType u32_dtype(xla::ifrt::DType::Kind::kU32);
  TF_ASSERT_OK_AND_ASSIGN(
      Value bitcast_val,
      xla_serving::Bitcast(std::move(value_orig), u32_dtype, xla::ifrt::Shape({1})));
  ASSERT_TRUE(bitcast_val.IsType<HostBuffer>());
  EXPECT_EQ(bitcast_val.Get<HostBuffer>().dtype(), u32_dtype);
  EXPECT_EQ(bitcast_val.Get<HostBuffer>().shape(), xla::ifrt::Shape({1}));

  // Invalid byte size returns error
  HostBuffer invalid_hb1(xla::ifrt::DType(xla::ifrt::DType::Kind::kU8),
                         xla::ifrt::Shape({4}));
  EXPECT_FALSE(
      xla_serving::Bitcast(Value(std::move(invalid_hb1)), u32_dtype,
                   xla::ifrt::Shape({2}))
          .ok());

  // Invalid negative dimension returns error
  HostBuffer invalid_hb2(xla::ifrt::DType(xla::ifrt::DType::Kind::kU8),
                         xla::ifrt::Shape({4}));
  EXPECT_FALSE(
      xla_serving::Bitcast(Value(std::move(invalid_hb2)), u32_dtype,
                   xla::ifrt::Shape({-1}))
          .ok());
}

TEST(ValueUtilsTest, ReshapeDeviceBuffer) {
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  xla::ifrt::DType dtype(xla::ifrt::DType::Kind::kF32);
  // Use [1, 6] -> [6] to ensure portable contiguous execution across CPU, GPU,
  // and TPU.
  xla::ifrt::Shape shape({1, 6});
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  xla::ifrt::Device* device = client->addressable_devices().at(0);
  xla::ifrt::ShardingRef sharding =
      xla::ifrt::SingleDeviceSharding::Create(device, xla::ifrt::MemoryKind());

  TF_ASSERT_OK_AND_ASSIGN(
      xla::ifrt::ArrayRef array,
      client->MakeArrayFromHostBuffer(
          data.data(), dtype, shape,
          /*byte_strides=*/std::nullopt, sharding, /*layout=*/nullptr,
          xla::ifrt::Client::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/{}));

  TF_ASSERT_OK_AND_ASSIGN(xla::ifrt::DeviceListRef device_list,
                          client->MakeDeviceList({device}));

  DeviceBuffer db = {
      .ifrt_client = client,
      .ifrt_array = array,
      .sharding = sharding,
      .device_list = device_list,
  };
  Value value_orig(std::move(db));

  TF_ASSERT_OK_AND_ASSIGN(
      Value reshaped_val,
      xla_serving::Reshape(std::move(value_orig), xla::ifrt::Shape({6})));
  ASSERT_TRUE(reshaped_val.IsType<DeviceBuffer>());
  const DeviceBuffer& reshaped_db = reshaped_val.Get<DeviceBuffer>();
  EXPECT_EQ(reshaped_db.ifrt_array->shape(), xla::ifrt::Shape({6}));
  EXPECT_EQ(reshaped_db.ifrt_array->dtype(), dtype);

  std::vector<float> out_data(6);
  TF_ASSERT_OK(
      reshaped_db.ifrt_array
          ->CopyToHostBuffer(out_data.data(), /*byte_strides=*/std::nullopt,
                             xla::ifrt::ArrayCopySemantics::kAlwaysCopy)
          .Await());
  EXPECT_EQ(out_data, data);

  // Size mismatch returns error
  DeviceBuffer invalid_db1 = {
      .ifrt_client = client,
      .ifrt_array = array,
      .sharding = sharding,
      .device_list = device_list,
  };
  EXPECT_FALSE(
      xla_serving::Reshape(Value(std::move(invalid_db1)), xla::ifrt::Shape({5})).ok());

  // Negative dimension returns error
  DeviceBuffer invalid_db2 = {
      .ifrt_client = client,
      .ifrt_array = array,
      .sharding = sharding,
      .device_list = device_list,
  };
  EXPECT_FALSE(
      xla_serving::Reshape(Value(std::move(invalid_db2)), xla::ifrt::Shape({-2, -3}))
          .ok());

  // Uninitialized DeviceBuffer returns error
  EXPECT_FALSE(
      xla_serving::Reshape(Value(DeviceBuffer{}), xla::ifrt::Shape({6})).ok());
}

TEST(ValueUtilsTest, BitcastDeviceBuffer) {
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  xla::ifrt::DType s32_dtype(xla::ifrt::DType::Kind::kS32);
  xla::ifrt::Shape shape({1});
  std::vector<int32_t> data = {-1};

  xla::ifrt::Device* device = client->addressable_devices().at(0);
  xla::ifrt::ShardingRef sharding =
      xla::ifrt::SingleDeviceSharding::Create(device, xla::ifrt::MemoryKind());

  TF_ASSERT_OK_AND_ASSIGN(
      xla::ifrt::ArrayRef array,
      client->MakeArrayFromHostBuffer(
          data.data(), s32_dtype, shape,
          /*byte_strides=*/std::nullopt, sharding, /*layout=*/nullptr,
          xla::ifrt::Client::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/{}));

  TF_ASSERT_OK_AND_ASSIGN(xla::ifrt::DeviceListRef device_list,
                          client->MakeDeviceList({device}));

  DeviceBuffer db = {
      .ifrt_client = client,
      .ifrt_array = array,
      .sharding = sharding,
      .device_list = device_list,
  };
  Value value_orig(std::move(db));

  xla::ifrt::DType u32_dtype(xla::ifrt::DType::Kind::kU32);
  TF_ASSERT_OK_AND_ASSIGN(
      Value bitcast_val,
      xla_serving::Bitcast(std::move(value_orig), u32_dtype, xla::ifrt::Shape({1})));
  ASSERT_TRUE(bitcast_val.IsType<DeviceBuffer>());
  const DeviceBuffer& bitcast_db = bitcast_val.Get<DeviceBuffer>();
  EXPECT_EQ(bitcast_db.ifrt_array->shape(), xla::ifrt::Shape({1}));
  EXPECT_EQ(bitcast_db.ifrt_array->dtype(), u32_dtype);

  std::vector<uint32_t> out_data(1);
  TF_ASSERT_OK(
      bitcast_db.ifrt_array
          ->CopyToHostBuffer(out_data.data(), /*byte_strides=*/std::nullopt,
                             xla::ifrt::ArrayCopySemantics::kAlwaysCopy)
          .Await());
  // Reinterpret int32_t -1 (two's complement 0xFFFFFFFF) as uint32_t
  // 4294967295u.
  EXPECT_EQ(out_data[0], static_cast<uint32_t>(-1));

  // Invalid byte size returns error (4 bytes -> 8 bytes)
  DeviceBuffer invalid_db1 = {
      .ifrt_client = client,
      .ifrt_array = array,
      .sharding = sharding,
      .device_list = device_list,
  };
  EXPECT_FALSE(
      xla_serving::Bitcast(Value(std::move(invalid_db1)), u32_dtype,
                   xla::ifrt::Shape({2}))
          .ok());

  // Negative dimension returns error
  DeviceBuffer invalid_db2 = {
      .ifrt_client = client,
      .ifrt_array = array,
      .sharding = sharding,
      .device_list = device_list,
  };
  EXPECT_FALSE(
      xla_serving::Bitcast(Value(std::move(invalid_db2)), u32_dtype,
                   xla::ifrt::Shape({-1}))
          .ok());

  // Uninitialized DeviceBuffer returns error
  EXPECT_FALSE(
      xla_serving::Bitcast(Value(DeviceBuffer{}), u32_dtype, xla::ifrt::Shape({1}))
          .ok());
}

TEST(ValueUtilsTest, NonBufferValueReturnsError) {
  EXPECT_FALSE(xla_serving::Reshape(Value(123), xla::ifrt::Shape({1})).ok());
  EXPECT_FALSE(
      xla_serving::Bitcast(Value(123), xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
                   xla::ifrt::Shape({1}))
          .ok());
}

}  // namespace
}  // namespace xla_serving
