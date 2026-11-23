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

#include "runtime/core/dynamic_dma_utils.h"  // from @xla_serving

#include <cstdint>
#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "runtime/core/h2d_transfer_executor.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/host_buffer_test_utils.h"  // from @xla_serving
#include "xla/layout_util.h"  // from @xla
#include "xla/pjrt/common_pjrt_client.h"  // from @xla
#include "xla/pjrt/pjrt_client.h"  // from @xla
#include "xla/pjrt/pjrt_layout.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/memory.h"  // from @xla
#include "xla/python/ifrt/rtti.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/ifrt/test_util.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_client.h"  // from @xla
#include "xla/tsl/lib/monitoring/cell_reader.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla
#include "tsl/platform/ml_dtypes.h"  // from @tsl
#include "tsl/platform/tstring.h"  // from @tsl

namespace xla_serving {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::tsl::monitoring::testing::CellReader;

// Runs on the CPU client, which stores an array as its elements in the order
// of its layout, without padding.
class DynamicDmaUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(client_, xla::ifrt::test_util::GetClient());
    auto* pjrt_compatible_client =
        xla::ifrt::dyn_cast_or_null<xla::ifrt::PjRtCompatibleClient>(
            client_.get());
    ASSERT_NE(pjrt_compatible_client, nullptr);
    common_client_ = dynamic_cast<xla::CommonPjRtClient*>(
        pjrt_compatible_client->pjrt_client());
    ASSERT_NE(common_client_, nullptr);
    ASSERT_FALSE(common_client_->devices().empty());
    TF_ASSERT_OK_AND_ASSIGN(
        memory_space_, common_client_->devices()[0]->default_memory_space());
  }

  // Plans the transfer of `input` to the default memory space of device 0.
  absl::StatusOr<DynamicDmaTransfer> Plan(
      const HostBuffer& input,
      const H2DTransferExecutor::StaticShape& static_shape) {
    return PlanDynamicDmaTransfer(input, xla::ifrt::MemoryKind(), static_shape,
                                  *common_client_, memory_space_);
  }

  std::shared_ptr<xla::ifrt::Client> client_;
  xla::CommonPjRtClient* common_client_ = nullptr;
  xla::PjRtMemorySpace* memory_space_ = nullptr;
};

TEST_F(DynamicDmaUtilsTest, PlansDeviceBufferOfStaticShapeAndCopyOfRows) {
  HostBuffer input = AsHostBuffer<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                                         xla::ifrt::Shape({2, 3}));

  TF_ASSERT_OK_AND_ASSIGN(DynamicDmaTransfer transfer,
                          Plan(input, {.shape = xla::ifrt::Shape({4, 3})}));

  EXPECT_EQ(transfer.device_shape.element_type(), xla::F32);
  EXPECT_THAT(transfer.device_shape.dimensions(), ElementsAre(4, 3));
  EXPECT_EQ(transfer.device_bytes, 4 * 3 * sizeof(float));
  // The rows are linearized in the layout of the device buffer.
  EXPECT_THAT(transfer.transfer_shape.dimensions(), ElementsAre(2, 3));
  EXPECT_EQ(transfer.transfer_shape.layout(), transfer.device_shape.layout());
  EXPECT_EQ(transfer.linearized_bytes, 2 * 3 * sizeof(float));
  EXPECT_EQ(transfer.dma_bytes, 2 * 3 * sizeof(float));
}

TEST_F(DynamicDmaUtilsTest, InputWithoutRowsHasNothingToCopy) {
  HostBuffer input = AsHostBuffer<float>({}, xla::ifrt::Shape({0, 3}));

  TF_ASSERT_OK_AND_ASSIGN(DynamicDmaTransfer transfer,
                          Plan(input, {.shape = xla::ifrt::Shape({4, 3})}));

  EXPECT_THAT(transfer.device_shape.dimensions(), ElementsAre(4, 3));
  EXPECT_EQ(transfer.device_bytes, 4 * 3 * sizeof(float));
  EXPECT_THAT(transfer.transfer_shape.dimensions(), ElementsAre(0, 3));
  EXPECT_EQ(transfer.linearized_bytes, 0);
  EXPECT_EQ(transfer.dma_bytes, 0);
}

TEST_F(DynamicDmaUtilsTest, DeviceBufferHasLayoutOfStaticShape) {
  HostBuffer input = AsHostBuffer<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                                         xla::ifrt::Shape({2, 3}));

  // Dimension 0 is the major-most dimension, as in the default layout.
  auto row_major = std::make_shared<const xla::PjRtLayout>(
      xla::LayoutUtil::MakeLayout({1, 0}));
  TF_ASSERT_OK_AND_ASSIGN(
      DynamicDmaTransfer transfer,
      Plan(input, {.shape = xla::ifrt::Shape({4, 3}), .layout = row_major}));
  EXPECT_THAT(transfer.device_shape.layout().minor_to_major(),
              ElementsAre(1, 0));

  // Dimension 0 is the minor-most dimension: the buffer starts with the first
  // column of all four rows, so the two rows of the input are not its front.
  auto column_major = std::make_shared<const xla::PjRtLayout>(
      xla::LayoutUtil::MakeLayout({0, 1}));
  EXPECT_THAT(
      Plan(input, {.shape = xla::ifrt::Shape({4, 3}), .layout = column_major}),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("major-most dimension")));
}

TEST_F(DynamicDmaUtilsTest, RejectsInputThatIsNotPrefixOfStaticShape) {
  HostBuffer input = AsHostBuffer<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                                         xla::ifrt::Shape({2, 3}));

  // Fewer rows than the input, another inner dimension, another rank.
  for (const xla::ifrt::Shape& static_shape :
       {xla::ifrt::Shape({1, 3}), xla::ifrt::Shape({4, 2}),
        xla::ifrt::Shape({12})}) {
    EXPECT_THAT(Plan(input, {.shape = static_shape}),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("not a dimension-0 prefix")))
        << static_shape;
  }
  // A scalar has no rows.
  EXPECT_THAT(Plan(AsHostBuffer<float>(1.0f), {.shape = xla::ifrt::Shape({})}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("not a dimension-0 prefix")));
}

TEST_F(DynamicDmaUtilsTest, RejectsUnsupportedDtypes) {
  tsl::tstring str_data[2] = {"hello", "world"};
  HostBuffer str_input(str_data,
                       xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                       xla::ifrt::Shape({2}));
  EXPECT_THAT(Plan(str_input, {.shape = xla::ifrt::Shape({4})}),
              StatusIs(absl::StatusCode::kUnimplemented, HasSubstr("dtype")));

  // The host stores an int4 element in a byte and the device packs it.
  HostBuffer int4_input =
      AsHostBuffer<tsl::int4, int32_t>({1, 2}, xla::ifrt::Shape({2}));
  EXPECT_THAT(Plan(int4_input, {.shape = xla::ifrt::Shape({4})}),
              StatusIs(absl::StatusCode::kUnimplemented, HasSubstr("dtype")));
}

TEST_F(DynamicDmaUtilsTest, TakesElementsWiderThan32BitsOffTpu) {
  // Only TPU splits an element that is wider than 32 bits.
  HostBuffer input = AsHostBuffer<int64_t>({1, 2}, xla::ifrt::Shape({2}));

  TF_ASSERT_OK_AND_ASSIGN(DynamicDmaTransfer transfer,
                          Plan(input, {.shape = xla::ifrt::Shape({4})}));

  EXPECT_EQ(transfer.device_bytes, 4 * sizeof(int64_t));
  EXPECT_EQ(transfer.dma_bytes, 2 * sizeof(int64_t));
}

TEST_F(DynamicDmaUtilsTest, RejectsNonDefaultMemoryKind) {
  HostBuffer input = AsHostBuffer<float>({1.0f, 2.0f}, xla::ifrt::Shape({2}));

  EXPECT_THAT(
      PlanDynamicDmaTransfer(input, xla::ifrt::MemoryKind("pinned_host"),
                             {.shape = xla::ifrt::Shape({4})}, *common_client_,
                             memory_space_),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("default memory kind")));
}

TEST_F(DynamicDmaUtilsTest, RejectsStaticShapeWithoutElements) {
  HostBuffer input = AsHostBuffer<float>({}, xla::ifrt::Shape({1, 0}));

  EXPECT_THAT(Plan(input, {.shape = xla::ifrt::Shape({4, 0})}),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("without elements")));
}

TEST(RecordDynamicDmaBytesTest, AddsBytesToCells) {
  CellReader<int64_t> dynamic_dma_bytes(
      "/xla_serving/input_transfer/dynamic_dma_bytes");

  RecordDynamicDmaBytes(/*allocated_bytes=*/48, /*transferred_bytes=*/24);
  RecordDynamicDmaBytes(/*allocated_bytes=*/16, /*transferred_bytes=*/0);

  EXPECT_EQ(dynamic_dma_bytes.Delta("allocated"), 64);
  EXPECT_EQ(dynamic_dma_bytes.Delta("transferred"), 24);
}

}  // namespace
}  // namespace xla_serving
