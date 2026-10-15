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

#include "runtime/core/ifrt_device_utils.h"  // from @xla_serving

#include <memory>
#include <optional>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "xla/python/ifrt/attribute_map.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/mock.h"  // from @xla
#include "xla/service/device_assignment.h"  // from @xla
#include "xla/tsl/platform/status_matchers.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla

namespace xla_serving {
namespace {
using ::testing::ElementsAre;
using ::testing::Return;
using ::testing::ReturnRef;

static constexpr int kNumReplicas = 1;
static constexpr int kNumCoresPerReplica = 2;
// Intentionally have more devices than kNumReplicas * kNumCoresPerReplica for
// testing purposes.
static constexpr int kNumDevices = 4;
static constexpr int kDeviceIdOffset = 8;

class IfrtDeviceUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mocked_devices_.reserve(kNumDevices);
    devices_.reserve(kNumDevices);
    for (int i = 0; i < kNumDevices; ++i) {
      mocked_devices_.push_back(std::make_unique<xla::ifrt::MockDevice>());
      ON_CALL(*mocked_devices_[i], Attributes())
          .WillByDefault(ReturnRef(device_attributes_maps_[i]));
      ON_CALL(*mocked_devices_[i], Id())
          .WillByDefault(Return(xla::ifrt::DeviceId(kDeviceIdOffset + i)));
      ON_CALL(client_, LookupDevice(xla::ifrt::DeviceId(kDeviceIdOffset + i)))
          .WillByDefault(Return(mocked_devices_[i].get()));

      devices_.push_back(mocked_devices_[i].get());
    };

    ON_CALL(client_, devices()).WillByDefault(Return(devices_));

    // Default use the last two devices.
    xla::DeviceAssignment assignment(kNumReplicas, kNumCoresPerReplica);
    assignment(0, 0) = kDeviceIdOffset + 2;
    assignment(0, 1) = kDeviceIdOffset + 3;

    ON_CALL(client_,
            GetDefaultDeviceAssignment(kNumReplicas, kNumCoresPerReplica))
        .WillByDefault(Return(assignment));
  }

  xla::ifrt::MockClient client_;
  std::vector<std::unique_ptr<xla::ifrt::MockDevice>> mocked_devices_;

  std::vector<xla::ifrt::Device*> devices_;
  std::vector<xla::ifrt::AttributeMap> device_attributes_maps_ = {
      xla::ifrt::AttributeMap(xla::ifrt::AttributeMap::Map{
          {"coords", xla::ifrt::AttributeMap::Int64ListValue({1, 0, 0})},
          {"core_on_chip", xla::ifrt::AttributeMap::Int64Value(0)}}),
      xla::ifrt::AttributeMap(xla::ifrt::AttributeMap::Map{
          {"coords", xla::ifrt::AttributeMap::Int64ListValue({1, 0, 0})},
          {"core_on_chip", xla::ifrt::AttributeMap::Int64Value(1)}}),
      xla::ifrt::AttributeMap(xla::ifrt::AttributeMap::Map{
          {"coords", xla::ifrt::AttributeMap::Int64ListValue({2, 0, 0})},
          {"core_on_chip", xla::ifrt::AttributeMap::Int64Value(0)}}),
      xla::ifrt::AttributeMap(xla::ifrt::AttributeMap::Map{
          {"coords", xla::ifrt::AttributeMap::Int64ListValue({2, 0, 0})},
          {"core_on_chip", xla::ifrt::AttributeMap::Int64Value(1)}}),
  };
};

TEST_F(IfrtDeviceUtilsTest, Basic) {
  std::vector<int> device_assignment_attr = {1, 0, 0, 1, 1, 0, 0, 0};
  TF_ASSERT_OK_AND_ASSIGN(
      auto devices_from_attribute,
      GetAssignedIfrtDevices(client_, kNumReplicas, kNumCoresPerReplica,
                             device_assignment_attr));
  EXPECT_THAT(devices_from_attribute, ElementsAre(devices_[1], devices_[0]));
}

TEST_F(IfrtDeviceUtilsTest, SeparateXCoordinates) {
  std::vector<int> device_assignment_attr = {1, 0, 0, 1, 2, 0, 0, 0};
  TF_ASSERT_OK_AND_ASSIGN(
      auto devices_from_attribute,
      GetAssignedIfrtDevices(client_, kNumReplicas, kNumCoresPerReplica,
                             device_assignment_attr));
  EXPECT_THAT(devices_from_attribute, ElementsAre(devices_[1], devices_[2]));
}

TEST_F(IfrtDeviceUtilsTest, EmptyDeviceAssignmentShallReturnDefault) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto devices_from_attribute,
      GetAssignedIfrtDevices(client_, kNumReplicas, kNumCoresPerReplica,
                             std::nullopt));
  EXPECT_THAT(devices_from_attribute, ElementsAre(devices_[2], devices_[3]));
}

TEST_F(IfrtDeviceUtilsTest, MismatchCoordinatesShallFail) {
  std::vector<int> device_assignment_attr = {1, 0, 0, 1, 3, 0, 0, 0};
  auto status = GetAssignedIfrtDevices(client_, 1, 2, device_assignment_attr);
  EXPECT_THAT(status,
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
}

}  // namespace

}  // namespace xla_serving
