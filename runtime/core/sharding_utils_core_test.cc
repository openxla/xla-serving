#define EIGEN_USE_THREADS

#include "third_party/neptune/runtime/core/sharding_utils_core.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/log/log.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/types/span.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "third_party/neptune/runtime/core/host_buffer.h"
#include "third_party/neptune/runtime/core/host_buffer_matcher.h"
#include "third_party/neptune/runtime/core/host_buffer_test_utils.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/dtype.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/shape.h"
#include "third_party/tensorflow/compiler/xla/tsl/lib/core/status_test_util.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/env.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/threadpool.h"

namespace jsv::sharding_internal {
namespace {
using ::testing::ElementsAre;

Eigen::ThreadPoolDevice CreateThreadPoolDevice() {
  constexpr int kMaxParallelism = 16;
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "Resharding", kMaxParallelism);

  Eigen::ThreadPoolDevice device(thread_pool->AsEigenThreadPool(),
                                 kMaxParallelism);
  return device;
}

TEST(XlaNDSplitterTest, NoSplits) {
  auto device = CreateThreadPoolDevice();

  const xla::ifrt::Shape input_shape({2, 2, 2});
  const std::vector<int32_t> num_splits = {1, 1, 1};
  const std::vector<int> paddings(num_splits.size(), 0);
  const int num_outputs = 1;
  auto input_host_buffer =
      AsHostBuffer<int32_t>({0, 1, 2, 3, 4, 5, 6, 7}, input_shape);

  std::vector<HostBuffer> output_host_buffers;
  output_host_buffers.resize(num_outputs);
  auto allocate_output_fn = [&](int i,
                                const xla::ifrt::Shape& output_slice_shape,
                                HostBuffer** tensor) {
    if (i < 0 || i >= output_host_buffers.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Index ", i, " out of range [0, ", output_host_buffers.size(), ")"));
    }
    output_host_buffers[i] = HostBuffer(
        xla::ifrt::DType(xla::ifrt::DType::Kind::kS32), output_slice_shape);
    *tensor = &output_host_buffers[i];
    return absl::OkStatus();
  };
  auto assign_or_copy_value_fn = [&](const HostBuffer& input) -> absl::Status {
    output_host_buffers[0] = input;
    return absl::OkStatus();
  };

  TF_ASSERT_OK_AND_ASSIGN(
      auto splitter, (XlaNDSplitter<Eigen::ThreadPoolDevice, int32_t>::Create(
                         num_splits, num_outputs, paddings,
                         /*has_paddings=*/false)));
  TF_ASSERT_OK(splitter.Split(&input_host_buffer, "test",
                              assign_or_copy_value_fn, allocate_output_fn,
                              device));

  EXPECT_THAT(output_host_buffers,
              ElementsAre(HostBufferEq(AsHostBuffer<int32_t>(
                  {0, 1, 2, 3, 4, 5, 6, 7}, xla::ifrt::Shape({2, 2, 2})))));
}

TEST(XlaNDSplitterTest, NoSplitsWithPadding) {
  auto device = CreateThreadPoolDevice();

  const xla::ifrt::Shape input_shape({2, 1, 1});
  const std::vector<int32_t> num_splits = {1, 1, 1};
  const std::vector<int> paddings = {0, 1, 1};
  const int num_outputs = 1;
  auto input_host_buffer = AsHostBuffer<int32_t>({0, 1}, input_shape);

  std::vector<HostBuffer> output_host_buffers;
  output_host_buffers.resize(num_outputs);
  auto allocate_output_fn = [&](int i,
                                const xla::ifrt::Shape& output_slice_shape,
                                HostBuffer** tensor) {
    if (i < 0 || i >= output_host_buffers.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Index ", i, " out of range [0, ", output_host_buffers.size(), "]"));
    }
    output_host_buffers[i] = HostBuffer(
        xla::ifrt::DType(xla::ifrt::DType::Kind::kS32), output_slice_shape);
    *tensor = &output_host_buffers[i];
    VLOG(2) << "output_slice_shape: " << output_slice_shape;
    return absl::OkStatus();
  };
  auto assign_or_copy_value_fn = [&](const HostBuffer& input) -> absl::Status {
    output_host_buffers[0] = input;
    return absl::OkStatus();
  };

  TF_ASSERT_OK_AND_ASSIGN(
      auto splitter, (XlaNDSplitter<Eigen::ThreadPoolDevice, int32_t>::Create(
                         num_splits, num_outputs, paddings,
                         /*has_paddings=*/true)));

  TF_ASSERT_OK(splitter.Split(&input_host_buffer, "test",
                              assign_or_copy_value_fn, allocate_output_fn,
                              device));

  EXPECT_THAT(output_host_buffers,
              ElementsAre(HostBufferEq(AsHostBuffer<int32_t>(
                  {0, 0, 0, 0, 1, 0, 0, 0}, xla::ifrt::Shape({2, 2, 2})))));
}

TEST(XlaNDSplitterTest, SplitNoPadding) {
  auto device = CreateThreadPoolDevice();

  const xla::ifrt::Shape input_shape({4, 4});
  const std::vector<int32_t> num_splits = {2, 2};
  const std::vector<int32_t> paddings(num_splits.size(), 0);
  const int num_outputs = 4;
  auto input_host_buffer = AsHostBuffer<int32_t>(
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, input_shape);

  std::vector<HostBuffer> output_host_buffers;
  output_host_buffers.resize(num_outputs);
  auto allocate_output_fn = [&](int i,
                                const xla::ifrt::Shape& output_slice_shape,
                                HostBuffer** tensor) {
    if (i < 0 || i >= output_host_buffers.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Index ", i, " out of range [0, ", output_host_buffers.size(), "]"));
    }
    output_host_buffers[i] = HostBuffer(
        xla::ifrt::DType(xla::ifrt::DType::Kind::kS32), output_slice_shape);
    *tensor = &output_host_buffers[i];
    LOG(INFO) << "output_slice_shape: " << output_slice_shape;
    return absl::OkStatus();
  };
  auto assign_or_copy_value_fn = [&](const HostBuffer& input) -> absl::Status {
    output_host_buffers[0] = input;
    return absl::OkStatus();
  };

  TF_ASSERT_OK_AND_ASSIGN(
      auto splitter, (XlaNDSplitter<Eigen::ThreadPoolDevice, int32_t>::Create(
                         num_splits, num_outputs, paddings,
                         /*has_paddings=*/true)));

  TF_ASSERT_OK(splitter.Split(&input_host_buffer, "test",
                              assign_or_copy_value_fn, allocate_output_fn,
                              device));

  EXPECT_THAT(output_host_buffers,
              ElementsAre(HostBufferEq(AsHostBuffer<int32_t>(
                              {0, 1, 4, 5}, xla::ifrt::Shape({2, 2}))),
                          HostBufferEq(AsHostBuffer<int32_t>(
                              {2, 3, 6, 7}, xla::ifrt::Shape({2, 2}))),
                          HostBufferEq(AsHostBuffer<int32_t>(
                              {8, 9, 12, 13}, xla::ifrt::Shape({2, 2}))),
                          HostBufferEq(AsHostBuffer<int32_t>(
                              {10, 11, 14, 15}, xla::ifrt::Shape({2, 2})))));
}

TEST(XlaNDSplitterTest, SplitPartialPadding) {
  auto device = CreateThreadPoolDevice();

  const xla::ifrt::Shape input_shape({3, 3});
  const std::vector<int32_t> num_splits = {2, 2};
  const std::vector<int32_t> paddings = {1, 1};
  const int num_outputs = 4;
  auto input_host_buffer =
      AsHostBuffer<int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8}, input_shape);

  std::vector<HostBuffer> output_host_buffers;
  output_host_buffers.resize(num_outputs);
  auto allocate_output_fn = [&](int i,
                                const xla::ifrt::Shape& output_slice_shape,
                                HostBuffer** tensor) {
    if (i < 0 || i >= output_host_buffers.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Index ", i, " out of range [0, ", output_host_buffers.size(), "]"));
    }
    output_host_buffers[i] = HostBuffer(
        xla::ifrt::DType(xla::ifrt::DType::Kind::kS32), output_slice_shape);
    *tensor = &output_host_buffers[i];
    LOG(INFO) << "output_slice_shape: " << output_slice_shape;
    return absl::OkStatus();
  };
  auto assign_or_copy_value_fn = [&](const HostBuffer& input) -> absl::Status {
    output_host_buffers[0] = input;
    return absl::OkStatus();
  };

  TF_ASSERT_OK_AND_ASSIGN(
      auto splitter, (XlaNDSplitter<Eigen::ThreadPoolDevice, int32_t>::Create(
                         num_splits, num_outputs, paddings,
                         /*has_paddings=*/true)));

  TF_ASSERT_OK(splitter.Split(&input_host_buffer, "test",
                              assign_or_copy_value_fn, allocate_output_fn,
                              device));

  EXPECT_THAT(output_host_buffers,
              ElementsAre(HostBufferEq(AsHostBuffer<int32_t>(
                              {0, 1, 3, 4}, xla::ifrt::Shape({2, 2}))),
                          HostBufferEq(AsHostBuffer<int32_t>(
                              {2, 0, 5, 0}, xla::ifrt::Shape({2, 2}))),
                          HostBufferEq(AsHostBuffer<int32_t>(
                              {6, 7, 0, 0}, xla::ifrt::Shape({2, 2}))),
                          HostBufferEq(AsHostBuffer<int32_t>(
                              {8, 0, 0, 0}, xla::ifrt::Shape({2, 2})))));
}

TEST(XlaNDSplitterTest, SplitCompletePadding) {
  auto device = CreateThreadPoolDevice();

  const xla::ifrt::Shape input_shape({2, 1});
  const std::vector<int32_t> num_splits = {2, 2};
  const std::vector<int32_t> paddings = {2, 3};
  const int num_outputs = 4;
  auto input_host_buffer = AsHostBuffer<int32_t>({0, 1}, input_shape);

  std::vector<HostBuffer> output_host_buffers;
  output_host_buffers.resize(num_outputs);
  auto allocate_output_fn = [&](int i,
                                const xla::ifrt::Shape& output_slice_shape,
                                HostBuffer** tensor) {
    if (i < 0 || i >= output_host_buffers.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Index ", i, " out of range [0, ", output_host_buffers.size(), "]"));
    }
    output_host_buffers[i] = HostBuffer(
        xla::ifrt::DType(xla::ifrt::DType::Kind::kS32), output_slice_shape);
    *tensor = &output_host_buffers[i];
    LOG(INFO) << "output_slice_shape: " << output_slice_shape;
    return absl::OkStatus();
  };
  auto assign_or_copy_value_fn = [&](const HostBuffer& input) -> absl::Status {
    output_host_buffers[0] = input;
    return absl::OkStatus();
  };

  TF_ASSERT_OK_AND_ASSIGN(
      auto splitter, (XlaNDSplitter<Eigen::ThreadPoolDevice, int32_t>::Create(
                         num_splits, num_outputs, paddings,
                         /*has_paddings=*/true)));

  TF_ASSERT_OK(splitter.Split(&input_host_buffer, "test",
                              assign_or_copy_value_fn, allocate_output_fn,
                              device));

  EXPECT_THAT(output_host_buffers,
              ElementsAre(HostBufferEq(AsHostBuffer<int32_t>(
                              {0, 0, 1, 0}, xla::ifrt::Shape({2, 2}))),
                          HostBufferEq(AsHostBuffer<int32_t>(
                              {0, 0, 0, 0}, xla::ifrt::Shape({2, 2}))),
                          HostBufferEq(AsHostBuffer<int32_t>(
                              {0, 0, 0, 0}, xla::ifrt::Shape({2, 2}))),
                          HostBufferEq(AsHostBuffer<int32_t>(
                              {0, 0, 0, 0}, xla::ifrt::Shape({2, 2})))));
}

TEST(XlaNDConcatenatorTest, NoConcats) {
  auto device = CreateThreadPoolDevice();

  const xla::ifrt::Shape input_shape({2, 2, 2});
  const xla::ifrt::Shape output_shape({2, 2, 2});
  const std::vector<int32_t> num_concats = {1, 1, 1};
  const std::vector<int> paddings(num_concats.size(), 0);
  int num_slices = 1;

  std::vector<HostBuffer> input_host_buffers = {
      AsHostBuffer<int32_t>({0, 1, 2, 3, 4, 5, 6, 7}, input_shape),
  };

  std::vector<HostBuffer> output_host_buffers;
  output_host_buffers.reserve(1);
  auto get_output_fn = [&]() {
    output_host_buffers.push_back(HostBuffer(
        xla::ifrt::DType(xla::ifrt::DType::Kind::kS32), output_shape));
    return &output_host_buffers.back();
  };
  auto assign_or_copy_value_fn = [&](const HostBuffer& input) -> absl::Status {
    output_host_buffers.push_back(input);
    return absl::OkStatus();
  };

  TF_ASSERT_OK_AND_ASSIGN(
      auto concatenator,
      (XlaNDConcatenator<Eigen::ThreadPoolDevice, int32_t>::Create(
          num_concats, num_slices, paddings,
          /*has_paddings=*/true)));

  TF_ASSERT_OK(concatenator.ComputeInternal(absl::MakeSpan(input_host_buffers),
                                            assign_or_copy_value_fn,
                                            get_output_fn, device));

  EXPECT_THAT(output_host_buffers,
              ElementsAre(HostBufferEq(AsHostBuffer<int32_t>(
                  {0, 1, 2, 3, 4, 5, 6, 7}, xla::ifrt::Shape({2, 2, 2})))));
}

TEST(XlaNDConcatenatorTest, ConcatNoPadding) {
  auto device = CreateThreadPoolDevice();

  const xla::ifrt::Shape input_shape({2, 2});
  const xla::ifrt::Shape output_shape({4, 4});
  const std::vector<int32_t> num_concats = {2, 2};
  const std::vector<int> paddings(num_concats.size(), 0);
  int num_slices = 4;
  std::vector<HostBuffer> input_host_buffers = {
      AsHostBuffer<int32_t>({0, 1, 2, 3}, input_shape),
      AsHostBuffer<int32_t>({4, 5, 6, 7}, input_shape),
      AsHostBuffer<int32_t>({8, 9, 10, 11}, input_shape),
      AsHostBuffer<int32_t>({12, 13, 14, 15}, input_shape),
  };

  std::vector<HostBuffer> output_host_buffers;
  output_host_buffers.reserve(1);
  auto get_output_fn = [&]() {
    output_host_buffers.push_back(HostBuffer(
        xla::ifrt::DType(xla::ifrt::DType::Kind::kS32), output_shape));
    return &output_host_buffers.back();
  };
  auto assign_or_copy_value_fn = [&](const HostBuffer& input) -> absl::Status {
    output_host_buffers.push_back(input);
    return absl::OkStatus();
  };

  TF_ASSERT_OK_AND_ASSIGN(
      auto concatenator,
      (XlaNDConcatenator<Eigen::ThreadPoolDevice, int32_t>::Create(
          num_concats, num_slices, paddings,
          /*has_paddings=*/true)));

  TF_ASSERT_OK(concatenator.ComputeInternal(absl::MakeSpan(input_host_buffers),
                                            assign_or_copy_value_fn,
                                            get_output_fn, device));
  EXPECT_THAT(output_host_buffers,
              ElementsAre(HostBufferEq(AsHostBuffer<int32_t>(
                  {0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15},
                  xla::ifrt::Shape({4, 4})))));
}

TEST(XlaNDConcatenatorTest, ConcatPartialPadding) {
  auto device = CreateThreadPoolDevice();

  const xla::ifrt::Shape input_shape({2, 2});
  const xla::ifrt::Shape output_shape({3, 3});
  const std::vector<int32_t> num_concats = {2, 2};
  const std::vector<int> paddings = {1, 1};
  int num_slices = 4;
  std::vector<HostBuffer> input_host_buffers = {
      AsHostBuffer<int32_t>({0, 1, 2, 3}, input_shape),
      AsHostBuffer<int32_t>({4, 5, 6, 7}, input_shape),
      AsHostBuffer<int32_t>({8, 9, 10, 11}, input_shape),
      AsHostBuffer<int32_t>({12, 13, 14, 15}, input_shape),
  };

  std::vector<HostBuffer> output_host_buffers;
  output_host_buffers.reserve(1);
  auto get_output_fn = [&]() {
    output_host_buffers.push_back(HostBuffer(
        xla::ifrt::DType(xla::ifrt::DType::Kind::kS32), output_shape));
    return &output_host_buffers.back();
  };
  auto assign_or_copy_value_fn = [&](const HostBuffer& input) -> absl::Status {
    output_host_buffers.push_back(input);
    return absl::OkStatus();
  };

  TF_ASSERT_OK_AND_ASSIGN(
      auto concatenator,
      (XlaNDConcatenator<Eigen::ThreadPoolDevice, int32_t>::Create(
          num_concats, num_slices, paddings,
          /*has_paddings=*/true)));

  TF_ASSERT_OK(concatenator.ComputeInternal(absl::MakeSpan(input_host_buffers),
                                            assign_or_copy_value_fn,
                                            get_output_fn, device));

  EXPECT_THAT(output_host_buffers,
              ElementsAre(HostBufferEq(AsHostBuffer<int32_t>(
                  {0, 1, 4, 2, 3, 6, 8, 9, 12}, xla::ifrt::Shape({3, 3})))));
}

TEST(XlaNDConcatenatorTest, ConcatCompletePadding) {
  auto device = CreateThreadPoolDevice();

  const xla::ifrt::Shape input_shape({2, 2});
  const xla::ifrt::Shape output_shape({2, 2});
  const std::vector<int32_t> num_concats = {2, 2};
  const std::vector<int> paddings = {2, 2};
  int num_slices = 4;

  std::vector<HostBuffer> input_host_buffers = {
      AsHostBuffer<int32_t>({0, 1, 2, 3}, input_shape),
      AsHostBuffer<int32_t>({4, 5, 6, 7}, input_shape),
      AsHostBuffer<int32_t>({8, 9, 10, 11}, input_shape),
      AsHostBuffer<int32_t>({12, 13, 14, 15}, input_shape),
  };

  std::vector<HostBuffer> output_host_buffers;
  output_host_buffers.reserve(1);
  auto get_output_fn = [&]() {
    output_host_buffers.push_back(HostBuffer(
        xla::ifrt::DType(xla::ifrt::DType::Kind::kS32), output_shape));
    return &output_host_buffers.back();
  };
  auto assign_or_copy_value_fn = [&](const HostBuffer& input) -> absl::Status {
    output_host_buffers.push_back(input);
    return absl::OkStatus();
  };

  TF_ASSERT_OK_AND_ASSIGN(
      auto concatenator,
      (XlaNDConcatenator<Eigen::ThreadPoolDevice, int32_t>::Create(
          num_concats, num_slices, paddings,
          /*has_paddings=*/true)));

  TF_ASSERT_OK(concatenator.ComputeInternal(absl::MakeSpan(input_host_buffers),
                                            assign_or_copy_value_fn,
                                            get_output_fn, device));

  EXPECT_THAT(output_host_buffers,
              ElementsAre(HostBufferEq(AsHostBuffer<int32_t>(
                  {0, 1, 2, 3}, xla::ifrt::Shape({2, 2})))));
}

}  // namespace
}  // namespace jsv::sharding_internal
