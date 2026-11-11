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

#include "runtime/core/streaming/stream.h"  // from @xla_serving

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "xla/tsl/platform/test_benchmark.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "Eigen/Core"  // IWYU pragma: keep
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/streaming/utils.h"  // from @xla_serving
#include "xla/executable_run_options.h"  // from @xla
#include "xla/ffi/api/c_api.h"  // from @xla
#include "xla/ffi/call_frame.h"  // from @xla
#include "xla/ffi/execution_state.h"  // from @xla
#include "xla/ffi/ffi.h"  // from @xla
#include "xla/ffi/ffi_registry.h"  // from @xla
#include "xla/ffi/invoke.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/stream_executor/device_address.h"  // from @xla
#include "xla/tsl/lib/core/status_test_util.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla

// IWYU pragma: no_include "Eigen/src/Core/util/ConfigureVectorization.h"

namespace xla_serving {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

using ResponseList =
    std::vector<std::pair<int64_t, std::vector<xla_serving::HostBuffer>>>;

struct RecordedCall {
  int64_t callback_id = -1;
  std::vector<std::string> names;
  ResponseList responses;
};

class RecordingStreamWorker : public StreamWorkerInterface {
 public:
  RecordingStreamWorker(std::string controller_address,
                        std::vector<RecordedCall>* calls)
      : StreamWorkerInterface(std::move(controller_address)), calls_(calls) {}

  absl::Status InvokeStreamCallback(int64_t callback_id,
                                    const std::vector<std::string>& names,
                                    const ResponseList& responses) override {
    if (calls_ != nullptr) {
      calls_->push_back({callback_id, names, responses});
    }
    return absl::OkStatus();
  }

 private:
  std::vector<RecordedCall>* calls_;
};

class StreamFfiTest : public ::testing::Test {
 protected:
  void SetUp() override {
    calls_.clear();
    GetGlobalStreamInterfaceFactory().RegisterWorker(
        [this](absl::string_view address)
            -> absl::StatusOr<std::unique_ptr<StreamWorkerInterface>> {
          return std::make_unique<RecordingStreamWorker>(std::string(address),
                                                         &calls_);
        });

    TF_ASSERT_OK_AND_ASSIGN(
        registration_, xla::ffi::FindHandler("xla_serving_stream", "Host"));

    xla::ffi::CallFrameBuilder::AttributesBuilder attrs;
    attrs.Insert("controller_address", "test_controller");
    xla::ffi::CallFrameBuilder instantiate_builder(/*num_args=*/0,
                                                   /*num_rets=*/0);
    instantiate_builder.AddAttributes(attrs.Build());
    xla::ffi::CallFrame instantiate_frame = instantiate_builder.Build();

    context_.run_id = xla::RunId{1};
    context_.state_context = {&state_[0], &state_[1], &state_[2]};
    TF_ASSERT_OK(xla::ffi::Invoke(
        xla::ffi::GetXlaFfiApi(), registration_.bundle.instantiate,
        instantiate_frame, context_, XLA_FFI_ExecutionStage_INSTANTIATE));
  }

  absl::Status InvokeExecute(xla::ffi::CallFrame& call_frame) {
    return xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(),
                            registration_.bundle.execute, call_frame, context_);
  }

  std::vector<RecordedCall> calls_;
  xla::ffi::HandlerRegistration registration_;
  std::array<xla::ffi::ExecutionState, 3> state_;
  xla::ffi::InvokeContext context_;
};

template <typename T>
std::vector<T> ReadFlat(const xla_serving::HostBuffer& buffer) {
  auto flat = buffer.flat<T>();
  return std::vector<T>(flat.data(),
                        flat.data() + buffer.shape().num_elements());
}

TEST_F(StreamFfiTest, UnbatchedValidStepIdCopiesAndAlignsBuffers) {
  // Use 3 floats (12 bytes) so the buffer size is not a multiple of
  // EIGEN_MAX_ALIGN_BYTES.
  std::vector<float> arg0 = {1.0f, 2.0f, 3.0f};
  std::vector<int32_t> arg1 = {10, 20};
  int32_t step_id = 42;

  xla::ffi::CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("callback_id", int64_t{77});
  attrs.Insert("output_names",
               absl::StrCat("out0,out1,", kStreamedResultBatchedStepIdName));

  xla::ffi::CallFrameBuilder builder(/*num_args=*/3, /*num_rets=*/1);
  builder.AddBufferArg(stream_executor::DeviceAddressBase(
                           arg0.data(), arg0.size() * sizeof(float)),
                       xla::PrimitiveType::F32, /*dims=*/{3});
  builder.AddBufferArg(stream_executor::DeviceAddressBase(
                           arg1.data(), arg1.size() * sizeof(int32_t)),
                       xla::PrimitiveType::S32, /*dims=*/{2});
  builder.AddBufferArg(
      stream_executor::DeviceAddressBase(&step_id, sizeof(int32_t)),
      xla::PrimitiveType::S32, /*dims=*/{});
  builder.AddTokenRet();
  builder.AddAttributes(attrs.Build());
  xla::ffi::CallFrame frame = builder.Build();

  TF_ASSERT_OK(InvokeExecute(frame));

  ASSERT_EQ(calls_.size(), 1);
  EXPECT_EQ(calls_[0].callback_id, 77);
  EXPECT_THAT(calls_[0].names, ElementsAre("out0", "out1"));
  ASSERT_EQ(calls_[0].responses.size(), 1);
  EXPECT_EQ(calls_[0].responses[0].first, 42);
  ASSERT_EQ(calls_[0].responses[0].second.size(), 2);

  const xla_serving::HostBuffer& hb0 = calls_[0].responses[0].second[0];
  const xla_serving::HostBuffer& hb1 = calls_[0].responses[0].second[1];
  EXPECT_TRUE(hb0.IsAligned(EIGEN_MAX_ALIGN_BYTES));
  EXPECT_TRUE(hb1.IsAligned(EIGEN_MAX_ALIGN_BYTES));
  EXPECT_EQ(hb0.shape(), xla::ifrt::Shape({3}));
  EXPECT_EQ(hb1.shape(), xla::ifrt::Shape({2}));
  EXPECT_THAT(ReadFlat<float>(hb0), ElementsAre(1.0f, 2.0f, 3.0f));
  EXPECT_THAT(ReadFlat<int32_t>(hb1), ElementsAre(10, 20));
}

TEST_F(StreamFfiTest, UnbatchedInvalidStepIdSkipsCallback) {
  std::vector<float> arg0 = {1.0f, 2.0f, 3.0f, 4.0f};
  int32_t step_id = -1;

  xla::ffi::CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("callback_id", int64_t{77});
  attrs.Insert("output_names",
               absl::StrCat("out0,", kStreamedResultBatchedStepIdName));

  xla::ffi::CallFrameBuilder builder(/*num_args=*/2, /*num_rets=*/1);
  builder.AddBufferArg(stream_executor::DeviceAddressBase(
                           arg0.data(), arg0.size() * sizeof(float)),
                       xla::PrimitiveType::F32, /*dims=*/{4});
  builder.AddBufferArg(
      stream_executor::DeviceAddressBase(&step_id, sizeof(int32_t)),
      xla::PrimitiveType::S32, /*dims=*/{});
  builder.AddTokenRet();
  builder.AddAttributes(attrs.Build());
  xla::ffi::CallFrame frame = builder.Build();

  TF_ASSERT_OK(InvokeExecute(frame));
  EXPECT_TRUE(calls_.empty());
}

TEST_F(StreamFfiTest, BatchedWithPaddingCopiesOnlyActiveSlicesAndAlignsEach) {
  // Batch size 6, inner dim 3 floats (12 bytes per row, so row 2 at byte offset
  // 24 is not 32/64-byte aligned in the input buffer).
  std::vector<float> arg0(6 * 3);
  std::iota(arg0.begin(), arg0.end(), 1.0f);
  std::vector<int32_t> step_ids = {10, 10, 20, -1, -1, -1};

  xla::ffi::CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("callback_id", int64_t{88});
  attrs.Insert("output_names",
               absl::StrCat("out0,", kStreamedResultBatchedStepIdName));

  xla::ffi::CallFrameBuilder builder(/*num_args=*/2, /*num_rets=*/1);
  builder.AddBufferArg(stream_executor::DeviceAddressBase(
                           arg0.data(), arg0.size() * sizeof(float)),
                       xla::PrimitiveType::F32, /*dims=*/{6, 3});
  builder.AddBufferArg(stream_executor::DeviceAddressBase(
                           step_ids.data(), step_ids.size() * sizeof(int32_t)),
                       xla::PrimitiveType::S32, /*dims=*/{6});
  builder.AddTokenRet();
  builder.AddAttributes(attrs.Build());
  xla::ffi::CallFrame frame = builder.Build();

  TF_ASSERT_OK(InvokeExecute(frame));

  ASSERT_EQ(calls_.size(), 1);
  EXPECT_EQ(calls_[0].callback_id, 88);
  EXPECT_THAT(calls_[0].names, ElementsAre("out0"));
  ASSERT_EQ(calls_[0].responses.size(), 2);

  EXPECT_EQ(calls_[0].responses[0].first, 10);
  ASSERT_EQ(calls_[0].responses[0].second.size(), 1);
  const xla_serving::HostBuffer& req10_buf = calls_[0].responses[0].second[0];
  EXPECT_TRUE(req10_buf.IsAligned(EIGEN_MAX_ALIGN_BYTES));
  EXPECT_EQ(req10_buf.shape(), xla::ifrt::Shape({2, 3}));
  EXPECT_EQ(req10_buf.total_storage_bytes(), 2 * 3 * sizeof(float));
  EXPECT_THAT(ReadFlat<float>(req10_buf),
              ElementsAre(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f));

  EXPECT_EQ(calls_[0].responses[1].first, 20);
  ASSERT_EQ(calls_[0].responses[1].second.size(), 1);
  const xla_serving::HostBuffer& req20_buf = calls_[0].responses[1].second[0];
  EXPECT_TRUE(req20_buf.IsAligned(EIGEN_MAX_ALIGN_BYTES));
  EXPECT_EQ(req20_buf.shape(), xla::ifrt::Shape({1, 3}));
  EXPECT_EQ(req20_buf.total_storage_bytes(), 1 * 3 * sizeof(float));
  EXPECT_THAT(ReadFlat<float>(req20_buf), ElementsAre(7.0f, 8.0f, 9.0f));
}

TEST_F(StreamFfiTest, BatchedAllPaddingSkipsCallback) {
  std::vector<float> arg0(4 * 8, 1.0f);
  std::vector<int32_t> step_ids = {-1, -1, -1, -1};

  xla::ffi::CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("callback_id", int64_t{99});
  attrs.Insert("output_names",
               absl::StrCat("out0,", kStreamedResultBatchedStepIdName));

  xla::ffi::CallFrameBuilder builder(/*num_args=*/2, /*num_rets=*/1);
  builder.AddBufferArg(stream_executor::DeviceAddressBase(
                           arg0.data(), arg0.size() * sizeof(float)),
                       xla::PrimitiveType::F32, /*dims=*/{4, 8});
  builder.AddBufferArg(stream_executor::DeviceAddressBase(
                           step_ids.data(), step_ids.size() * sizeof(int32_t)),
                       xla::PrimitiveType::S32, /*dims=*/{4});
  builder.AddTokenRet();
  builder.AddAttributes(attrs.Build());
  xla::ffi::CallFrame frame = builder.Build();

  TF_ASSERT_OK(InvokeExecute(frame));
  EXPECT_TRUE(calls_.empty());
}

TEST_F(StreamFfiTest, ValidatesStepIdAndDTypesBeforeAllocation) {
  std::vector<float> arg0(4, 1.0f);
  float bad_step_id = 1.0f;

  auto make_one_arg_attrs = []() {
    xla::ffi::CallFrameBuilder::AttributesBuilder attrs;
    attrs.Insert("callback_id", int64_t{55});
    attrs.Insert("output_names",
                 absl::StrCat("out0,", kStreamedResultBatchedStepIdName));
    return attrs.Build();
  };

  // 1. Non-S32 step_id_arg fails with InternalError.
  {
    xla::ffi::CallFrameBuilder builder(/*num_args=*/2, /*num_rets=*/1);
    builder.AddBufferArg(stream_executor::DeviceAddressBase(
                             arg0.data(), arg0.size() * sizeof(float)),
                         xla::PrimitiveType::F32, /*dims=*/{4});
    builder.AddBufferArg(
        stream_executor::DeviceAddressBase(&bad_step_id, sizeof(float)),
        xla::PrimitiveType::F32, /*dims=*/{});
    builder.AddTokenRet();
    builder.AddAttributes(make_one_arg_attrs());
    xla::ffi::CallFrame frame = builder.Build();

    EXPECT_THAT(InvokeExecute(frame),
                StatusIs(absl::StatusCode::kInternal,
                         HasSubstr("Expected step ID argument to be S32")));
    EXPECT_TRUE(calls_.empty());
  }

  // 2. Rank-2 step_id_arg fails with InternalError.
  {
    std::vector<int32_t> rank2_step_ids = {1, 1, 1, 1};
    xla::ffi::CallFrameBuilder builder(/*num_args=*/2, /*num_rets=*/1);
    builder.AddBufferArg(stream_executor::DeviceAddressBase(
                             arg0.data(), arg0.size() * sizeof(float)),
                         xla::PrimitiveType::F32, /*dims=*/{4});
    builder.AddBufferArg(
        stream_executor::DeviceAddressBase(
            rank2_step_ids.data(), rank2_step_ids.size() * sizeof(int32_t)),
        xla::PrimitiveType::S32, /*dims=*/{2, 2});
    builder.AddTokenRet();
    builder.AddAttributes(make_one_arg_attrs());
    xla::ffi::CallFrame frame = builder.Build();

    EXPECT_THAT(
        InvokeExecute(frame),
        StatusIs(absl::StatusCode::kInternal,
                 HasSubstr("Expected step ID argument to be a 0D or 1D")));
    EXPECT_TRUE(calls_.empty());
  }

  // 3. Unsupported user argument dtype (TOKEN) fails with InvalidArgumentError
  // even when step_id == -1.
  {
    int32_t step_id = -1;
    xla::ffi::CallFrameBuilder::AttributesBuilder two_arg_attrs;
    two_arg_attrs.Insert("callback_id", int64_t{55});
    two_arg_attrs.Insert(
        "output_names",
        absl::StrCat("out0,out1,", kStreamedResultBatchedStepIdName));

    xla::ffi::CallFrameBuilder builder(/*num_args=*/3, /*num_rets=*/1);
    builder.AddBufferArg(stream_executor::DeviceAddressBase(
                             arg0.data(), arg0.size() * sizeof(float)),
                         xla::PrimitiveType::F32, /*dims=*/{4});
    builder.AddTokenArg();
    builder.AddBufferArg(
        stream_executor::DeviceAddressBase(&step_id, sizeof(int32_t)),
        xla::PrimitiveType::S32, /*dims=*/{});
    builder.AddTokenRet();
    builder.AddAttributes(two_arg_attrs.Build());
    xla::ffi::CallFrame frame = builder.Build();

    EXPECT_THAT(InvokeExecute(frame),
                StatusIs(absl::StatusCode::kInvalidArgument));
    EXPECT_TRUE(calls_.empty());
  }

  // 4. Batched 1D step_id_arg with mismatched user argument batch dimension
  // fails with InvalidArgumentError even when all step_ids == -1.
  {
    std::vector<int32_t> step_ids = {-1, -1};
    xla::ffi::CallFrameBuilder builder(/*num_args=*/2, /*num_rets=*/1);
    builder.AddBufferArg(stream_executor::DeviceAddressBase(
                             arg0.data(), arg0.size() * sizeof(float)),
                         xla::PrimitiveType::F32, /*dims=*/{4});
    builder.AddBufferArg(
        stream_executor::DeviceAddressBase(step_ids.data(),
                                           step_ids.size() * sizeof(int32_t)),
        xla::PrimitiveType::S32, /*dims=*/{2});
    builder.AddTokenRet();
    builder.AddAttributes(make_one_arg_attrs());
    xla::ffi::CallFrame frame = builder.Build();

    EXPECT_THAT(InvokeExecute(frame),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("batch dimension")));
    EXPECT_TRUE(calls_.empty());
  }
}

//===----------------------------------------------------------------------===//
// Benchmarks
//===----------------------------------------------------------------------===//

class NoOpStreamWorker : public StreamWorkerInterface {
 public:
  using StreamWorkerInterface::StreamWorkerInterface;

  absl::Status InvokeStreamCallback(int64_t callback_id,
                                    const std::vector<std::string>& names,
                                    const ResponseList& responses) override {
    benchmark::DoNotOptimize(callback_id);
    benchmark::DoNotOptimize(names);
    benchmark::DoNotOptimize(responses);
    return absl::OkStatus();
  }
};

struct BenchmarkHarness {
  xla::ffi::HandlerRegistration registration;
  std::array<xla::ffi::ExecutionState, 3> state;
  xla::ffi::InvokeContext context;

  BenchmarkHarness() {
    GetGlobalStreamInterfaceFactory().RegisterWorker(
        [](absl::string_view address)
            -> absl::StatusOr<std::unique_ptr<StreamWorkerInterface>> {
          return std::make_unique<NoOpStreamWorker>(std::string(address));
        });

    auto reg = xla::ffi::FindHandler("xla_serving_stream", "Host");
    CHECK_OK(reg.status());
    registration = *reg;

    xla::ffi::CallFrameBuilder::AttributesBuilder attrs;
    attrs.Insert("controller_address", "bench_controller");
    xla::ffi::CallFrameBuilder instantiate_builder(/*num_args=*/0,
                                                   /*num_rets=*/0);
    instantiate_builder.AddAttributes(attrs.Build());
    xla::ffi::CallFrame instantiate_frame = instantiate_builder.Build();

    context.run_id = xla::RunId{1};
    context.state_context = {&state[0], &state[1], &state[2]};
    CHECK_OK(xla::ffi::Invoke(
        xla::ffi::GetXlaFfiApi(), registration.bundle.instantiate,
        instantiate_frame, context, XLA_FFI_ExecutionStage_INSTANTIATE));
  }
};

void BM_StreamBatchedWithPadding(benchmark::State& state) {
  constexpr int64_t kBatchSize = 64;
  const int64_t active_rows = state.range(0);
  const int64_t floats_per_row = state.range(1);

  BenchmarkHarness harness;
  std::vector<float> tensor0(kBatchSize * floats_per_row, 1.0f);
  std::vector<float> tensor1(kBatchSize * floats_per_row, 2.0f);
  std::vector<int32_t> step_ids(kBatchSize, -1);
  for (int64_t i = 0; i < active_rows; ++i) {
    step_ids[i] = static_cast<int32_t>(100 + i);
  }

  xla::ffi::CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("callback_id", int64_t{1});
  attrs.Insert("output_names",
               absl::StrCat("out0,out1,", kStreamedResultBatchedStepIdName));

  xla::ffi::CallFrameBuilder builder(/*num_args=*/3, /*num_rets=*/1);
  builder.AddBufferArg(stream_executor::DeviceAddressBase(
                           tensor0.data(), tensor0.size() * sizeof(float)),
                       xla::PrimitiveType::F32,
                       /*dims=*/{kBatchSize, floats_per_row});
  builder.AddBufferArg(stream_executor::DeviceAddressBase(
                           tensor1.data(), tensor1.size() * sizeof(float)),
                       xla::PrimitiveType::F32,
                       /*dims=*/{kBatchSize, floats_per_row});
  builder.AddBufferArg(stream_executor::DeviceAddressBase(
                           step_ids.data(), step_ids.size() * sizeof(int32_t)),
                       xla::PrimitiveType::S32, /*dims=*/{kBatchSize});
  builder.AddTokenRet();
  builder.AddAttributes(attrs.Build());
  xla::ffi::CallFrame frame = builder.Build();

  for (auto s : state) {
    CHECK_OK(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(),
                              harness.registration.bundle.execute, frame,
                              harness.context));
  }
}

BENCHMARK(BM_StreamBatchedWithPadding)
    ->ArgPair(1, 1024)
    ->ArgPair(4, 1024)
    ->ArgPair(16, 1024)
    ->ArgPair(64, 1024)
    ->ArgPair(1, 16384)
    ->ArgPair(4, 16384)
    ->ArgPair(8, 16384)
    ->ArgPair(16, 16384)
    ->ArgPair(64, 16384);

void BM_StreamWarmupAllPadded(benchmark::State& state) {
  constexpr int64_t kBatchSize = 64;
  const int64_t floats_per_row = state.range(0);

  BenchmarkHarness harness;
  std::vector<float> tensor0(kBatchSize * floats_per_row, 1.0f);
  std::vector<float> tensor1(kBatchSize * floats_per_row, 2.0f);
  std::vector<int32_t> step_ids(kBatchSize, -1);

  xla::ffi::CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("callback_id", int64_t{1});
  attrs.Insert("output_names",
               absl::StrCat("out0,out1,", kStreamedResultBatchedStepIdName));

  xla::ffi::CallFrameBuilder builder(/*num_args=*/3, /*num_rets=*/1);
  builder.AddBufferArg(stream_executor::DeviceAddressBase(
                           tensor0.data(), tensor0.size() * sizeof(float)),
                       xla::PrimitiveType::F32,
                       /*dims=*/{kBatchSize, floats_per_row});
  builder.AddBufferArg(stream_executor::DeviceAddressBase(
                           tensor1.data(), tensor1.size() * sizeof(float)),
                       xla::PrimitiveType::F32,
                       /*dims=*/{kBatchSize, floats_per_row});
  builder.AddBufferArg(stream_executor::DeviceAddressBase(
                           step_ids.data(), step_ids.size() * sizeof(int32_t)),
                       xla::PrimitiveType::S32, /*dims=*/{kBatchSize});
  builder.AddTokenRet();
  builder.AddAttributes(attrs.Build());
  xla::ffi::CallFrame frame = builder.Build();

  for (auto s : state) {
    CHECK_OK(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(),
                              harness.registration.bundle.execute, frame,
                              harness.context));
  }
}

BENCHMARK(BM_StreamWarmupAllPadded)->Arg(1024)->Arg(16384);

void BM_StreamUnbatched(benchmark::State& state) {
  const int64_t num_floats = state.range(0);

  BenchmarkHarness harness;
  std::vector<float> tensor0(num_floats, 1.0f);
  std::vector<float> tensor1(num_floats, 2.0f);
  int32_t step_id = 100;

  xla::ffi::CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("callback_id", int64_t{1});
  attrs.Insert("output_names",
               absl::StrCat("out0,out1,", kStreamedResultBatchedStepIdName));

  xla::ffi::CallFrameBuilder builder(/*num_args=*/3, /*num_rets=*/1);
  builder.AddBufferArg(stream_executor::DeviceAddressBase(
                           tensor0.data(), tensor0.size() * sizeof(float)),
                       xla::PrimitiveType::F32, /*dims=*/{num_floats});
  builder.AddBufferArg(stream_executor::DeviceAddressBase(
                           tensor1.data(), tensor1.size() * sizeof(float)),
                       xla::PrimitiveType::F32, /*dims=*/{num_floats});
  builder.AddBufferArg(
      stream_executor::DeviceAddressBase(&step_id, sizeof(int32_t)),
      xla::PrimitiveType::S32, /*dims=*/{});
  builder.AddTokenRet();
  builder.AddAttributes(attrs.Build());
  xla::ffi::CallFrame frame = builder.Build();

  for (auto s : state) {
    CHECK_OK(xla::ffi::Invoke(xla::ffi::GetXlaFfiApi(),
                              harness.registration.bundle.execute, frame,
                              harness.context));
  }
}

BENCHMARK(BM_StreamUnbatched)->Arg(16384)->Arg(262144);

}  // namespace
}  // namespace xla_serving
