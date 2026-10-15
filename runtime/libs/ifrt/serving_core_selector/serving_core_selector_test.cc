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

#include "runtime/libs/ifrt/serving_core_selector/serving_core_selector.h"  // from @xla_serving

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "xla/tsl/framework/serving_device_selector.h"  // from @xla
#include "xla/tsl/framework/test_util/mock_serving_device_selector.h"  // from @xla
#include "xla/tsl/lib/monitoring/cell_reader.h"  // from @xla

namespace xla_serving {
namespace {

using ::tsl::monitoring::testing::CellReader;

class ServingCoreSelectorTest : public ::testing::Test {
 protected:
  ServingCoreSelectorTest() {
    core_selector_ = std::make_unique<ServingCoreSelector>(
        &serving_device_selector_, num_cores_,
        /*record_core_selection_by_fingerprint=*/true);
    core_selection_count_reader_ = std::make_unique<CellReader<int64_t>>(
        "/xla_serving/runtime/core_selection_count");
    core_selection_count_by_device_id_only_reader_ =
        std::make_unique<CellReader<int64_t>>(
            "/xla_serving/runtime/core_selection_count_by_device_id_only");
  }

  tsl::test_util::MockServingDeviceSelector serving_device_selector_;
  size_t num_cores_ = 2;
  std::unique_ptr<ServingCoreSelector> core_selector_;
  std::unique_ptr<CellReader<int64_t>> core_selection_count_reader_;
  std::unique_ptr<CellReader<int64_t>>
      core_selection_count_by_device_id_only_reader_;
};

TEST_F(ServingCoreSelectorTest, Basic) {
  const std::string program_fingerprint = "111111";
  // Set up the mock serving device selector to return device 1.
  const int kSelectedDeviceIndex = 1;
  EXPECT_CALL(serving_device_selector_, ReserveDevice(program_fingerprint))
      .WillOnce([this](::testing::Unused) {
        return tsl::DeviceReservation(kSelectedDeviceIndex,
                                      &serving_device_selector_);
      });
  // Warm up each core first.
  for (size_t i = 0; i < num_cores_; ++i) {
    EXPECT_EQ(core_selector_->ReserveDevice(program_fingerprint).device_index(),
              i);
    EXPECT_EQ(core_selection_count_reader_->Delta(program_fingerprint,
                                                 absl::StrCat(i)),
              1);
    EXPECT_EQ(
        core_selection_count_by_device_id_only_reader_->Delta(absl::StrCat(i)),
        1);
  }
  // After warmup, core selection should be delegated to
  // `serving_device_selector_`.
  tsl::DeviceReservation reservation =
      core_selector_->ReserveDevice(program_fingerprint);
  EXPECT_EQ(reservation.device_index(), kSelectedDeviceIndex);
  EXPECT_EQ(core_selection_count_reader_->Delta(
                program_fingerprint, absl::StrCat(kSelectedDeviceIndex)),
            1);
}

TEST_F(ServingCoreSelectorTest, MultiplePrograms) {
  const std::string program_fingerprint1 = "111111";
  const std::string program_fingerprint2 = "222222";
  // Warm up each core first.
  for (size_t i = 0; i < num_cores_; ++i) {
    EXPECT_EQ(
        core_selector_->ReserveDevice(program_fingerprint1).device_index(), i);
    EXPECT_EQ(core_selection_count_reader_->Delta(program_fingerprint1,
                                                 absl::StrCat(i)),
              1);
    EXPECT_EQ(
        core_selection_count_by_device_id_only_reader_->Delta(absl::StrCat(i)),
        1);
    EXPECT_EQ(
        core_selector_->ReserveDevice(program_fingerprint2).device_index(), i);
    EXPECT_EQ(core_selection_count_reader_->Delta(program_fingerprint2,
                                                 absl::StrCat(i)),
              1);
    EXPECT_EQ(
        core_selection_count_by_device_id_only_reader_->Delta(absl::StrCat(i)),
        1);
  }
  // After warmup, core selection should be delegated to
  // serving_device_selector_.
  EXPECT_CALL(serving_device_selector_, ReserveDevice(program_fingerprint1))
      .WillOnce([this](::testing::Unused) {
        return tsl::DeviceReservation(0, &serving_device_selector_);
      });
  EXPECT_CALL(serving_device_selector_, ReserveDevice(program_fingerprint2))
      .WillOnce([this](::testing::Unused) {
        return tsl::DeviceReservation(1, &serving_device_selector_);
      });
  EXPECT_EQ(core_selector_->ReserveDevice(program_fingerprint2).device_index(),
            1);
  EXPECT_EQ(core_selection_count_reader_->Delta(program_fingerprint2, "1"), 1);
  EXPECT_EQ(core_selection_count_by_device_id_only_reader_->Delta("1"), 1);
  EXPECT_EQ(core_selector_->ReserveDevice(program_fingerprint1).device_index(),
            0);
  EXPECT_EQ(core_selection_count_reader_->Delta(program_fingerprint1, "0"), 1);
  EXPECT_EQ(core_selection_count_by_device_id_only_reader_->Delta("0"), 1);
}

TEST_F(ServingCoreSelectorTest, FingerprintCounterNotRecordedByDefault) {
  ServingCoreSelector core_selector(&serving_device_selector_, num_cores_);
  // A raw binary fingerprint, like the ones used in production.
  const std::string program_fingerprint("\xde\xad\xbe\xef\x00\xff", 6);
  for (size_t i = 0; i < num_cores_; ++i) {
    EXPECT_EQ(core_selector.ReserveDevice(program_fingerprint).device_index(),
              i);
    EXPECT_EQ(core_selection_count_reader_->Delta(program_fingerprint,
                                                  absl::StrCat(i)),
              0);
    EXPECT_EQ(
        core_selection_count_by_device_id_only_reader_->Delta(absl::StrCat(i)),
        1);
  }
}

}  // namespace
}  // namespace xla_serving
