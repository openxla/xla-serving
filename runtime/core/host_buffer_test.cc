#include "third_party/neptune/runtime/core/host_buffer.h"

#include <cstdint>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/benchmark.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/log/check.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/types/span.h"
#include "third_party/neptune/runtime/core/allocator.h"
#include "third_party/neptune/runtime/core/default_allocator.h"
#include "third_party/neptune/runtime/core/host_buffer_matcher.h"
#include "third_party/neptune/runtime/core/host_buffer_test_utils.h"
#include "third_party/neptune/runtime/core/proto/host_buffer.proto.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/dtype.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/shape.h"
#include "third_party/tensorflow/compiler/xla/tsl/lib/core/status_test_util.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/env.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/threadpool.h"
#include "third_party/tensorflow/tsl/platform/bfloat16.h"
#include "third_party/tensorflow/tsl/platform/ml_dtypes.h"
#include "third_party/tensorflow/tsl/platform/tstring.h"

namespace jsv {

class HostBufferSplitEasyCasesTest : public ::testing::Test {
 protected:
  void SetUp() override { allocator_ = DefaultAllocator::GetInstance(); }

  Allocator* allocator_;

  absl::Status SplitEasyCases(HostBuffer* host_buffer,
                              absl::Span<const int64_t> sizes,
                              std::vector<HostBuffer>& output,
                              bool& easy_cases_done) {
    return host_buffer->SplitEasyCases(allocator_, sizes, output,
                                       easy_cases_done);
  }
};

namespace {

using ::testing::ElementsAre;
using ::testing::status::StatusIs;

TEST(HostBufferTest, NullPtrAllocator) {
  HostBuffer host_buffer(xla::ifrt::DType(xla::ifrt::DType::Kind::kU32),
                         xla::ifrt::Shape({2, 2}), nullptr);
  EXPECT_EQ(host_buffer.dtype(),
            xla::ifrt::DType(xla::ifrt::DType::Kind::kU32));
  EXPECT_EQ(host_buffer.shape(), xla::ifrt::Shape({2, 2}));
}

// Type parameterized test for testing HostBuffer mutation and storage of data
// type int8_t, int16_t, int32_t, float, double.
template <typename T>
class HostBufferMutationAndStorageTest : public ::testing::Test {};

TYPED_TEST_SUITE_P(HostBufferMutationAndStorageTest);

TYPED_TEST_P(HostBufferMutationAndStorageTest, CanMutate) {
  using T = TypeParam;
  std::vector<T> original_data = {1, 2, 3, 4};
  T mutated_data = static_cast<T>(127);
  std::vector<T> expected_data = {1, mutated_data, 3, 4};

  HostBuffer host_buffer =
      AsHostBuffer<T>(original_data, xla::ifrt::Shape({2, 2}));
  host_buffer.flat<T>()(1) = mutated_data;
  EXPECT_THAT(host_buffer, HostBufferEq(AsHostBuffer<T>(
                               expected_data, xla::ifrt::Shape({2, 2}))));
}

TYPED_TEST_P(HostBufferMutationAndStorageTest, TotalStorageBytes) {
  using T = TypeParam;
  std::vector<T> original_data = {1, 2, 3, 4};
  HostBuffer host_buffer =
      AsHostBuffer<T>(original_data, xla::ifrt::Shape({2, 2}));
  EXPECT_EQ(host_buffer.total_storage_bytes(),
            original_data.size() * sizeof(T));
}

REGISTER_TYPED_TEST_SUITE_P(HostBufferMutationAndStorageTest, CanMutate,
                            TotalStorageBytes);

using HostBufferTestTypes =
    ::testing::Types<int8_t, int16_t, int32_t, float, double>;

INSTANTIATE_TYPED_TEST_SUITE_P(HostBufferMutationAndStorageTests,
                               HostBufferMutationAndStorageTest,
                               HostBufferTestTypes);

TEST(HostBufferTest, BoolCanMutate) {
  HostBuffer host_buffer =
      AsHostBuffer<bool>({true, false, true, false}, xla::ifrt::Shape({2, 2}));
  host_buffer.flat<bool>()(1) = true;
  EXPECT_THAT(host_buffer,
              HostBufferEq(AsHostBuffer<bool>({true, true, true, false},
                                              xla::ifrt::Shape({2, 2}))));
}

TEST(HostBufferTest, Int4CanMutate) {
  HostBuffer host_buffer =
      AsHostBuffer<tsl::int4, int>({1, 2, 3, 1}, xla::ifrt::Shape({2, 2}));

  host_buffer.flat<tsl::int4>()(1) = tsl::int4(3);
  EXPECT_THAT(host_buffer, HostBufferEq(AsHostBuffer<tsl::int4, int>(
                               {1, 3, 3, 1}, xla::ifrt::Shape({2, 2}))));
}

TEST(HostBufferTest, Bfloat16CanMutate) {
  HostBuffer host_buffer = AsHostBuffer<tsl::bfloat16, float>(
      {1.0f, 2.0f, 3.0f, 1.0f}, xla::ifrt::Shape({2, 2}));

  host_buffer.flat<tsl::bfloat16>()(1) = tsl::bfloat16(3.0f);
  EXPECT_THAT(host_buffer,
              HostBufferEq(AsHostBuffer<tsl::bfloat16, float>(
                  {1.0f, 3.0f, 3.0f, 1.0f}, xla::ifrt::Shape({2, 2}))));
}

TEST(HostBufferTest, StringCanMutate) {
  std::string long_str;
  for (int i = 0; i < 1000000; ++i) {
    long_str += "long";
  }
  std::vector<HostBuffer> inputs;
  tsl::tstring str_data[6] = {"aaa", "bbb", "ccc", "ddd", "eee", long_str};
  HostBuffer host_buffer(str_data,
                         xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                         xla::ifrt::Shape({2, 3}));
  host_buffer.flat<tsl::tstring>()(1) = long_str;
  EXPECT_EQ(host_buffer.flat<tsl::tstring>()(1), long_str);
}

TEST(HostBufferTest, TotalStorageBytesForInt4) {
  HostBuffer host_buffer =
      AsHostBuffer<tsl::int4, int>({1, 2}, xla::ifrt::Shape({2}));
  ASSERT_EQ(host_buffer.total_storage_bytes(), 2);
}

TEST(HostBufferTest, TotalStorageBytesForBfloat16) {
  HostBuffer host_buffer =
      AsHostBuffer<tsl::bfloat16, float>({1.0f, 2.0f}, xla::ifrt::Shape({2}));
  // sizeof(tsl::bfloat16) is 2.
  EXPECT_EQ(host_buffer.total_storage_bytes(), 4);
}

TEST(HostBufferTest, Copy) {
  HostBuffer original =
      AsHostBuffer<float>({1.0, 2.0, 3.0}, xla::ifrt::Shape({3}));
  HostBuffer copy = original;
  EXPECT_THAT(copy, HostBufferEq(original));
}

TEST(HostBufferTest, StringCopy) {
  tsl::tstring str_data[3] = {"aaa", "bbb", "ccc"};
  HostBuffer original(str_data,
                         xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                         xla::ifrt::Shape({3}));
  HostBuffer copy = original;
  EXPECT_THAT(copy, HostBufferEq(original));
}

TEST(HostBufferTest, CopyIsAliveAfterOriginalBufferIsDestroyed) {
  HostBuffer copy;
  {
    HostBuffer original =
        AsHostBuffer<bool>({true, false, true}, xla::ifrt::Shape({3}));
    copy = original;
  }
  EXPECT_THAT(copy, HostBufferEq(AsHostBuffer<bool>({true, false, true},
                                                    xla::ifrt::Shape({3}))));
}

TEST(HostBufferTest, StringCopyIsAliveAfterOriginalBufferIsDestroyed) {
  HostBuffer copy;
  tsl::tstring str_data[3] = {"aaa", "bbb", "ccc"};
  {
    HostBuffer original(str_data,
      xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
      xla::ifrt::Shape({3}));
    copy = original;
  }
  HostBuffer expected(str_data,
    xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
    xla::ifrt::Shape({3}));
  EXPECT_THAT(copy, HostBufferEq(expected));
}

TEST(HostBufferTest, CopyAlsoMutates) {
  HostBuffer original =
      AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({2, 2}));
  HostBuffer copy = original;
  EXPECT_THAT(copy, HostBufferEq(original));

  original.flat<int32_t>()(1) = 200;
  // Verify that the copied buffer is also mutated after the mutation of the
  // original buffer.
  for (const HostBuffer& buffer : {original, copy}) {
    EXPECT_THAT(buffer, HostBufferEq(AsHostBuffer<int32_t>(
                            {1, 200, 3, 4}, xla::ifrt::Shape({2, 2}))));
  }

  copy.flat<int32_t>()(3) = 201;
  // Verify that the original buffer is also mutated after the mutation of the
  // copied buffer.
  for (const HostBuffer& buffer : {original, copy}) {
    EXPECT_THAT(buffer, HostBufferEq(AsHostBuffer<int32_t>(
                            {1, 200, 3, 201}, xla::ifrt::Shape({2, 2}))));
  }
}

TEST(HostBufferTest, StringCopyAlsoMutates) {
  HostBuffer original = AsHostBuffer<tsl::tstring, std::string>(
      {"aaa", "bbb", "ccc", "ddd"}, xla::ifrt::Shape({2, 2}));
  HostBuffer copy = original;
  EXPECT_THAT(copy, HostBufferEq(original));

  original.flat<tsl::tstring>()(1) = "changed";
  // Verify that the copied buffer is also mutated after the mutation of the
  // original buffer.
  for (const HostBuffer& buffer : {original, copy}) {
    EXPECT_THAT(buffer, HostBufferEq(AsHostBuffer<tsl::tstring, std::string>(
                            {"aaa", "changed", "ccc", "ddd"},
                            xla::ifrt::Shape({2, 2}))));
  }

  copy.flat<tsl::tstring>()(3) = "new_changed";
  // Verify that the original buffer is also mutated after the mutation of the
  // copied buffer.
  for (const HostBuffer& buffer : {original, copy}) {
    EXPECT_THAT(buffer, HostBufferEq(AsHostBuffer<tsl::tstring, std::string>(
                            {"aaa", "changed", "ccc", "new_changed"},
                            xla::ifrt::Shape({2, 2}))));
  }
}

TEST(HostBufferTest, Move) {
  HostBuffer original =
      AsHostBuffer<float>({1.0, 2.0, 3.0}, xla::ifrt::Shape({3}));
  int original_ref_count = original.host_array()->NumRef();
  HostBuffer move = std::move(original);
  EXPECT_THAT(move, HostBufferEq(AsHostBuffer<float>({1.0, 2.0, 3.0},
                                                     xla::ifrt::Shape({3}))));
  // Verify that the original buffer does not hold a valid host array after
  // move.
  EXPECT_EQ(move.host_array()->NumRef(), original_ref_count);
}

TEST(HostBufferTest, StringMove) {
  HostBuffer original = AsHostBuffer<tsl::tstring, std::string>(
      {"aaa", "bbb", "ccc"}, xla::ifrt::Shape({3}));
  int original_ref_count = original.string_data().ref_count();
  HostBuffer move = std::move(original);
  EXPECT_THAT(move, HostBufferEq(AsHostBuffer<tsl::tstring, std::string>(
                        {"aaa", "bbb", "ccc"}, xla::ifrt::Shape({3}))));
  // Verify that the original buffer does not hold a valid host array after
  // move.
  EXPECT_EQ(move.string_data().ref_count(), original_ref_count);
}

TEST(HostBufferTest, MoveIsAliveAfterOriginalBufferIsDestroyed) {
  HostBuffer move;
  {
    HostBuffer original =
        AsHostBuffer<double>({1.0, 2.0, 3.0}, xla::ifrt::Shape({1, 3, 1}));
    move = std::move(original);
  }
  EXPECT_THAT(move, HostBufferEq(AsHostBuffer<double>(
                        {1.0, 2.0, 3.0}, xla::ifrt::Shape({1, 3, 1}))));
}

class HostBufferManipulationTest : public ::testing::Test {
 protected:
  void SetUp() override { allocator_ = DefaultAllocator::GetInstance(); }

  Allocator* allocator_;
};

TEST_F(HostBufferManipulationTest, SliceStringBuffer) {
  std::string long_str;
  for (int i = 0; i < 1000000; ++i) {
    long_str += "long";
  }
  std::vector<HostBuffer> inputs;
  tsl::tstring str_data[6] = {"aaa", long_str, "ccc", "ddd", "eee", long_str};
  HostBuffer host_buffer(str_data,
                         xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                         xla::ifrt::Shape({2, 3}));

  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer first,
      host_buffer.SliceAlongFirstDimension(/*start=*/0, /*end=*/1, allocator_));
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer second,
      host_buffer.SliceAlongFirstDimension(/*start=*/1, /*end=*/2, allocator_));

  tsl::tstring expected_first_strings[3] = {"aaa", long_str, "ccc"};
  tsl::tstring expected_second_strings[3] = {"ddd", "eee", long_str};
  HostBuffer expected_first(expected_first_strings,
                            xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                            xla::ifrt::Shape({1, 3}));
  HostBuffer expected_second(expected_second_strings,
                             xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                             xla::ifrt::Shape({1, 3}));
  EXPECT_THAT(first, HostBufferEq(expected_first));
  EXPECT_THAT(second, HostBufferEq(expected_second));
}

TEST_F(HostBufferManipulationTest, Slice2DBuffer) {
  HostBuffer host_buffer = AsHostBuffer<int32_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      xla::ifrt::Shape({4, 4}));
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer sliced_host_buffer,
      host_buffer.SliceAlongFirstDimension(/*start=*/0, /*end=*/1, allocator_));
  EXPECT_THAT(sliced_host_buffer, HostBufferEq(AsHostBuffer<int32_t>(
                                      {1, 2, 3, 4}, xla::ifrt::Shape({1, 4}))));
}

TEST_F(HostBufferManipulationTest, Slice3DBufferForOneRow) {
  HostBuffer host_buffer = AsHostBuffer<int32_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      xla::ifrt::Shape({4, 2, 2}));
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer sliced_host_buffer,
      host_buffer.SliceAlongFirstDimension(/*start=*/0, /*end=*/1, allocator_));
  EXPECT_THAT(sliced_host_buffer,
              HostBufferEq(AsHostBuffer<int32_t>({1, 2, 3, 4},
                                                 xla::ifrt::Shape({1, 2, 2}))));
}

TEST_F(HostBufferManipulationTest, Slice3DBufferForMultipleRwos) {
  HostBuffer host_buffer = AsHostBuffer<int32_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      xla::ifrt::Shape({4, 2, 2}));
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer sliced_host_buffer,
      host_buffer.SliceAlongFirstDimension(/*start=*/1, /*end=*/3, allocator_));
  EXPECT_THAT(sliced_host_buffer,
              HostBufferEq(AsHostBuffer<int32_t>({5, 6, 7, 8, 9, 10, 11, 12},
                                                 xla::ifrt::Shape({2, 2, 2}))));
}

TEST_F(HostBufferManipulationTest,
       SlicedBufferLivesAfterOriginalBufferIsDestroyed) {
  HostBuffer sliced_host_buffer;
  {
    HostBuffer host_buffer = AsHostBuffer<int32_t>(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
        xla::ifrt::Shape({4, 4}));
    TF_ASSERT_OK_AND_ASSIGN(sliced_host_buffer,
                            host_buffer.SliceAlongFirstDimension(
                                /*start=*/0, /*end=*/1, allocator_));
  }

  EXPECT_THAT(sliced_host_buffer, HostBufferEq(AsHostBuffer<int32_t>(
                                      {1, 2, 3, 4}, xla::ifrt::Shape({1, 4}))));
}

TEST_F(HostBufferManipulationTest, SliceFailedOnScalarBuffer) {
  HostBuffer host_buffer = AsHostBuffer<int32_t>(1);

  auto sliced_host_buffer =
      host_buffer.SliceAlongFirstDimension(/*start=*/0, /*end=*/1, allocator_);

  EXPECT_THAT(sliced_host_buffer, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(HostBufferManipulationTest, SliceFailedStartLessThanZero) {
  HostBuffer host_buffer = AsHostBuffer<int32_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      xla::ifrt::Shape({4, 4}));

  auto sliced_host_buffer =
      host_buffer.SliceAlongFirstDimension(/*start=*/-1, /*end=*/1, allocator_);

  EXPECT_THAT(sliced_host_buffer, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(HostBufferManipulationTest, SliceFailedEndLessThanZero) {
  HostBuffer host_buffer = AsHostBuffer<int32_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      xla::ifrt::Shape({4, 4}));

  auto sliced_host_buffer =
      host_buffer.SliceAlongFirstDimension(/*start=*/0, /*end=*/-1, allocator_);

  EXPECT_THAT(sliced_host_buffer, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(HostBufferManipulationTest, SliceFailedStartLargerThanEnd) {
  HostBuffer host_buffer = AsHostBuffer<int32_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      xla::ifrt::Shape({4, 4}));

  auto sliced_host_buffer =
      host_buffer.SliceAlongFirstDimension(/*start=*/1, /*end=*/0, allocator_);

  EXPECT_THAT(sliced_host_buffer, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(HostBufferManipulationTest, SliceEmptyBuffer) {
  HostBuffer host_buffer(/*base=*/nullptr,
                         xla::ifrt::DType(xla::ifrt::DType::Kind::kU32),
                         xla::ifrt::Shape({0}));

  TF_ASSERT_OK_AND_ASSIGN(
      auto sliced_host_buffer,
      host_buffer.SliceAlongFirstDimension(/*start=*/0, /*end=*/0, allocator_));

  EXPECT_THAT(
      sliced_host_buffer,
      HostBufferEq(HostBuffer(xla::ifrt::DType(xla::ifrt::DType::Kind::kU32),
                              xla::ifrt::Shape({0}), allocator_)));
}

TEST_F(HostBufferManipulationTest, StringConcatAndSplit) {
  std::string long_str;
  for (int i = 0; i < 100000; ++i) {
    long_str += "long";
  }
  tsl::tstring str_data0[2] = {"aaa", long_str};
  tsl::tstring str_data1[4] = {"ccc", "ddd", long_str, "fff"};
  tsl::tstring str_data2[4] = {"ggg", long_str, "iii", "jjjj"};
  tsl::tstring str_data3[10] = {"aaa", long_str, "ccc",    "ddd", long_str,
                                "fff", "ggg",    long_str, "iii", "jjjj"};
  HostBuffer host_buffer0(str_data0,
                          xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                          xla::ifrt::Shape({1, 2}));
  HostBuffer host_buffer1(str_data1,
                          xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                          xla::ifrt::Shape({2, 2}));
  HostBuffer host_buffer2(str_data2,
                          xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                          xla::ifrt::Shape({2, 2}));
  HostBuffer expected_host_buffer(
      str_data3, xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
      xla::ifrt::Shape({5, 2}));
  std::vector<HostBuffer> inputs;
  inputs.push_back(host_buffer0);
  inputs.push_back(host_buffer1);
  inputs.push_back(host_buffer2);
  TF_ASSERT_OK_AND_ASSIGN(auto concat_output,
                          Concat(nullptr, inputs, allocator_));
  EXPECT_THAT(concat_output, HostBufferEq(expected_host_buffer));
  TF_ASSERT_OK_AND_ASSIGN(auto outputs, concat_output.SplitAlongFirstDimension(
                                            {1, 2, 2}, allocator_));
  EXPECT_THAT(outputs, ElementsAre(HostBufferEq(host_buffer0),
                                   HostBufferEq(host_buffer1),
                                   HostBufferEq(host_buffer2)));
}

TEST_F(HostBufferManipulationTest, SplitAlongFirstDimensionTrivial) {
  HostBuffer host_buffer = AsHostBuffer<int32_t>(
      {1, 2, 3, 4, 5, 6, 7, 8}, xla::ifrt::Shape({4, 2}));
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<HostBuffer> split_host_buffers,
      host_buffer.SplitAlongFirstDimension({4}, allocator_));

  EXPECT_THAT(split_host_buffers, ElementsAre(HostBufferEq(host_buffer)));
}

TEST_F(HostBufferManipulationTest, SplitAlongFirstDimension) {
  std::vector<HostBuffer> split_host_buffers;
  {
    HostBuffer host_buffer = AsHostBuffer<int32_t>(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
        xla::ifrt::Shape({4, 4}));
    TF_ASSERT_OK_AND_ASSIGN(
        split_host_buffers,
        host_buffer.SplitAlongFirstDimension({3, 1}, allocator_));
  }

  EXPECT_THAT(split_host_buffers,
              ElementsAre(HostBufferEq(AsHostBuffer<int32_t>(
                              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                              xla::ifrt::Shape({3, 4}))),
                          HostBufferEq(AsHostBuffer<int32_t>(
                              {13, 14, 15, 16}, xla::ifrt::Shape({1, 4})))));
}

TEST_F(HostBufferManipulationTest, SplitSmallTypeAlongFirstDimension) {
  std::vector<HostBuffer> split_host_buffers;
  {
    HostBuffer host_buffer = AsHostBuffer<tsl::int4, int>(
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
        xla::ifrt::Shape({4, 4}));
    TF_ASSERT_OK_AND_ASSIGN(
        split_host_buffers,
        host_buffer.SplitAlongFirstDimension({3, 1}, allocator_));
  }

  for (const HostBuffer& split_host_buffer : split_host_buffers) {
    ASSERT_EQ(split_host_buffer.dtype(),
              xla::ifrt::DType(xla::ifrt::DType::Kind::kS4));
  }
  EXPECT_THAT(split_host_buffers,
              ElementsAre(HostBufferEq(AsHostBuffer<tsl::int4, int>(
                              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                              xla::ifrt::Shape({3, 4}))),
                          HostBufferEq(AsHostBuffer<tsl::int4, int>(
                              {12, 13, 14, 15}, xla::ifrt::Shape({1, 4})))));
}

TEST_F(HostBufferManipulationTest, Reshape) {
  HostBuffer host_buffer = AsHostBuffer<int32_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      xla::ifrt::Shape({4, 4}));
  auto reshaped_tensor =
      host_buffer.shaped<int32_t, 2>((std::vector<int64_t>{2, 8}));
  TF_ASSERT_OK(reshaped_tensor);
  EXPECT_EQ(reshaped_tensor->dimension(0), 2);
  EXPECT_EQ(reshaped_tensor->dimension(1), 8);
}

TEST_F(HostBufferManipulationTest, ConcatEmptyBuffers) {
  std::vector<HostBuffer> inputs;
  inputs.push_back(AsHostBuffer<int32_t>({}, xla::ifrt::Shape({2, 0})));
  inputs.push_back(AsHostBuffer<int32_t>({}, xla::ifrt::Shape({4, 0})));
  inputs.push_back(AsHostBuffer<int32_t>({}, xla::ifrt::Shape({6, 0})));
  TF_ASSERT_OK_AND_ASSIGN(auto output, Concat(nullptr, inputs, allocator_));

  EXPECT_THAT(output, HostBufferEq(AsHostBuffer<int32_t>(
                          {}, xla::ifrt::Shape({12, 0}))));
}

TEST_F(HostBufferManipulationTest, ConcatSingleBufferReturnsSameBuffer) {
  HostBuffer original_buffer =
      AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({2, 2}));
  std::vector<HostBuffer> inputs;
  inputs.push_back(original_buffer);
  TF_ASSERT_OK_AND_ASSIGN(auto output, Concat(nullptr, inputs, allocator_));

  EXPECT_THAT(output, HostBufferEq(original_buffer));
  EXPECT_EQ(output.base(), original_buffer.base());
}

TEST_F(HostBufferManipulationTest, ConcatSingleDimensionBuffers) {
  std::vector<HostBuffer> inputs;
  inputs.push_back(AsHostBuffer<int32_t>({1, 2}, xla::ifrt::Shape({2})));
  inputs.push_back(AsHostBuffer<int32_t>({3, 4, 5, 6}, xla::ifrt::Shape({4})));
  inputs.push_back(
      AsHostBuffer<int32_t>({7, 8, 9, 10, 11, 12}, xla::ifrt::Shape({6})));
  TF_ASSERT_OK_AND_ASSIGN(auto output, Concat(nullptr, inputs, allocator_));

  EXPECT_THAT(output, HostBufferEq(AsHostBuffer<int32_t>(
                          {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                          xla::ifrt::Shape({12}))));
}

TEST_F(HostBufferManipulationTest, ConcatWithZeroBatchDimension) {
  std::vector<HostBuffer> inputs;
  inputs.push_back(AsHostBuffer<int32_t>({}, xla::ifrt::Shape({0, 2, 2})));
  inputs.push_back(
      AsHostBuffer<int32_t>({3, 4, 5, 6}, xla::ifrt::Shape({1, 2, 2})));
  inputs.push_back(AsHostBuffer<int32_t>({1, 2, 3, 4, 5, 6, 7, 8},
                                         xla::ifrt::Shape({2, 2, 2})));
  TF_ASSERT_OK_AND_ASSIGN(auto output, Concat(nullptr, inputs, allocator_));

  EXPECT_THAT(output, HostBufferEq(AsHostBuffer<int32_t>(
                          {3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8},
                          xla::ifrt::Shape({3, 2, 2}))));
}

TEST_F(HostBufferManipulationTest, Concat) {
  std::vector<HostBuffer> inputs;
  inputs.push_back(
      AsHostBuffer<int32_t>({3, 4, 5, 6}, xla::ifrt::Shape({1, 2, 2})));
  inputs.push_back(AsHostBuffer<int32_t>({1, 2, 3, 4, 5, 6, 7, 8},
                                         xla::ifrt::Shape({2, 2, 2})));
  TF_ASSERT_OK_AND_ASSIGN(auto output, Concat(nullptr, inputs, allocator_));

  EXPECT_THAT(output, HostBufferEq(AsHostBuffer<int32_t>(
                          {3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8},
                          xla::ifrt::Shape({3, 2, 2}))));
}

TEST_F(HostBufferManipulationTest, ConcatLargeBuffer) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "concat_test", 5);
  auto gen_data = [](absl::Span<const int64_t> shape,
                     std::vector<int32_t>* data) -> HostBuffer {
    xla::ifrt::Shape input_shape(shape);
    data->resize(input_shape.num_elements());
    std::iota(data->begin(), data->end(), 0);
    HostBuffer host_buffer = AsHostBuffer<int32_t>(*data, input_shape);
    return host_buffer;
  };
  std::vector<int32_t> data1, data2, data3;
  std::vector<HostBuffer> inputs;
  std::vector<int32_t> output_data;
  int batch_size = 0;
  for (int i = 0; i < 3; ++i) {
    std::vector<int32_t> data;
    inputs.push_back(gen_data({i, 7777, 9999}, &data));
    batch_size += i;
    output_data.insert(output_data.end(), data.begin(), data.end());
  }

  xla::ifrt::Shape output_shape({batch_size, 7777, 9999});

  TF_ASSERT_OK_AND_ASSIGN(auto output,
                          Concat(&thread_pool, inputs, allocator_));
  EXPECT_THAT(output,
              HostBufferEq(AsHostBuffer<int32_t>(output_data, output_shape)));
}

TEST_F(HostBufferManipulationTest, ConcatLargeBufferWithSmallType) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "concat_test", 5);
  auto gen_data = [](absl::Span<const int64_t> shape,
                     std::vector<int32_t>* data) -> HostBuffer {
    xla::ifrt::Shape input_shape(shape);
    data->resize(input_shape.num_elements());
    std::iota(data->begin(), data->end(), 0);
    HostBuffer host_buffer =
        AsHostBuffer<tsl::int4, int32_t>(*data, input_shape);
    return host_buffer;
  };
  std::vector<int32_t> data1, data2, data3;
  std::vector<HostBuffer> inputs;
  std::vector<int32_t> output_data;
  int batch_size = 0;
  for (int i = 0; i < 3; ++i) {
    std::vector<int32_t> data;
    inputs.push_back(gen_data({i, 7777, 9999}, &data));
    batch_size += i;
    output_data.insert(output_data.end(), data.begin(), data.end());
  }

  xla::ifrt::Shape output_shape({batch_size, 7777, 9999});

  TF_ASSERT_OK_AND_ASSIGN(auto output,
                          Concat(&thread_pool, inputs, allocator_));
  EXPECT_THAT(output, HostBufferEq(AsHostBuffer<tsl::int4, int32_t>(
                          output_data, output_shape)));
}

TEST_F(HostBufferManipulationTest, ConcatLargeBufferWithFloatType) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "concat_test", 5);
  auto gen_data = [](absl::Span<const int64_t> shape,
                     std::vector<float>* data) -> HostBuffer {
    xla::ifrt::Shape input_shape(shape);
    data->resize(input_shape.num_elements());
    std::iota(data->begin(), data->end(), 0);
    HostBuffer host_buffer = AsHostBuffer<float>(*data, input_shape);
    return host_buffer;
  };
  std::vector<float> data1, data2, data3;
  std::vector<HostBuffer> inputs;
  std::vector<float> output_data;
  int batch_size = 0;
  for (int i = 0; i < 3; ++i) {
    std::vector<float> data;
    inputs.push_back(gen_data({i, 777, 999}, &data));
    batch_size += i;
    output_data.insert(output_data.end(), data.begin(), data.end());
  }

  xla::ifrt::Shape output_shape({batch_size, 777, 999});

  TF_ASSERT_OK_AND_ASSIGN(auto output,
                          Concat(&thread_pool, inputs, allocator_));
  EXPECT_THAT(output,
              HostBufferEq(AsHostBuffer<float>(output_data, output_shape)));
}

void BM_Concat(benchmark::State& state) {
  std::vector<HostBuffer> inputs;
  auto gen_buffer = [](absl::Span<const int64_t> shape) -> HostBuffer {
    xla::ifrt::Shape input_shape(shape);
    std::vector<int32_t> data;
    data.resize(input_shape.num_elements());
    std::iota(data.begin(), data.end(), 0);
    HostBuffer host_buffer = AsHostBuffer<int32_t>(data, input_shape);
    return host_buffer;
  };
  for (int i = 0; i < 3; ++i) {
    inputs.push_back(gen_buffer({i, 777, 999}));
  }
  for (auto s : state) {
    CHECK_OK(Concat(nullptr, inputs, DefaultAllocator::GetInstance()));
  }
}

BENCHMARK(BM_Concat);

void BM_ConcatParallel(benchmark::State& state) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "concat_test", 5);
  std::vector<HostBuffer> inputs;
  auto gen_buffer = [](absl::Span<const int64_t> shape) -> HostBuffer {
    xla::ifrt::Shape input_shape(shape);
    std::vector<int32_t> data;
    data.resize(input_shape.num_elements());
    std::iota(data.begin(), data.end(), 0);
    HostBuffer host_buffer = AsHostBuffer<int32_t>(data, input_shape);
    return host_buffer;
  };
  for (int i = 0; i < 3; ++i) {
    inputs.push_back(gen_buffer({i, 777, 999}));
  }
  for (auto s : state) {
    CHECK_OK(Concat(&thread_pool, inputs, DefaultAllocator::GetInstance()));
  }
}

BENCHMARK(BM_ConcatParallel);

TEST_F(HostBufferManipulationTest, ConcatSmallType) {
  std::vector<HostBuffer> inputs;
  inputs.push_back(AsHostBuffer<tsl::int4, int32_t>(
      {3, 4, 5, 6}, xla::ifrt::Shape({1, 2, 2})));
  inputs.push_back(AsHostBuffer<tsl::int4, int32_t>(
      {1, 2, 3, 4, 5, 6, 7, 8}, xla::ifrt::Shape({2, 2, 2})));
  TF_ASSERT_OK_AND_ASSIGN(auto output, Concat(nullptr, inputs, allocator_));
  EXPECT_THAT(output.dtype(), xla::ifrt::DType(xla::ifrt::DType::Kind::kS4));
  EXPECT_THAT(output, HostBufferEq(AsHostBuffer<tsl::int4, int32_t>(
                          {3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 7, 8},
                          xla::ifrt::Shape({3, 2, 2}))));
}

TEST_F(HostBufferManipulationTest, ConcatString) {
  tsl::tstring str_data1[6] = {"aaa", "bbb", "ccc", "ddd", "eee", "fff"};
  tsl::tstring str_data2[4] = {"ggg", "hhh", "iii", "jjjj"};
  tsl::tstring str_data3[10] = {"aaa", "bbb", "ccc", "ddd", "eee",
                                "fff", "ggg", "hhh", "iii", "jjjj"};
  HostBuffer host_buffer1(str_data1,
                          xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                          xla::ifrt::Shape({3, 2}));
  HostBuffer host_buffer2(str_data2,
                          xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                          xla::ifrt::Shape({2, 2}));
  HostBuffer expected_host_buffer(
      str_data3, xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
      xla::ifrt::Shape({5, 2}));
  std::vector<HostBuffer> inputs;
  inputs.push_back(host_buffer1);
  inputs.push_back(host_buffer2);
  TF_ASSERT_OK_AND_ASSIGN(auto output, Concat(nullptr, inputs, allocator_));
  EXPECT_THAT(output, HostBufferEq(expected_host_buffer));
}

TEST_F(HostBufferManipulationTest, ConcatFailedOnEmptyInput) {
  EXPECT_THAT(Concat(nullptr, {}, allocator_),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(HostBufferManipulationTest, ConcatFailedOnDifferentRankInputs) {
  std::vector<HostBuffer> inputs;
  inputs.push_back(AsHostBuffer<int32_t>({1, 2}, xla::ifrt::Shape({1, 2})));
  inputs.push_back(AsHostBuffer<int32_t>({3, 4, 5, 6}, xla::ifrt::Shape({4})));
  inputs.push_back(AsHostBuffer<int32_t>({7, 8, 9, 10, 11, 12},
                                         xla::ifrt::Shape({1, 2, 3})));
  EXPECT_THAT(Concat(nullptr, inputs, allocator_),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(HostBufferManipulationTest, ConcatFailedOnDifferentDimensionsInputs) {
  std::vector<HostBuffer> inputs;
  inputs.push_back(AsHostBuffer<int32_t>({1, 2}, xla::ifrt::Shape({1, 2})));
  inputs.push_back(
      AsHostBuffer<int32_t>({3, 4, 5, 6}, xla::ifrt::Shape({1, 4})));
  EXPECT_THAT(Concat(nullptr, inputs, allocator_),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(HostBufferSplitEasyCasesTest, SplitEasyCasesSizeOne) {
  HostBuffer host_buffer =
      AsHostBuffer<int32_t>({1, 2, 3, 4, 5, 6, 7, 8}, xla::ifrt::Shape({4, 2}));
  std::vector<HostBuffer> output;
  bool easy_cases_done = false;
  ASSERT_OK(SplitEasyCases(&host_buffer, {4}, output, easy_cases_done));
  EXPECT_TRUE(easy_cases_done);
  EXPECT_THAT(output, ElementsAre(HostBufferEq(host_buffer)));
  EXPECT_EQ(output[0].host_array()->raw_data(),
            host_buffer.host_array()->raw_data());
}

TEST_F(HostBufferSplitEasyCasesTest, SplitEasyCasesAligned) {
  // Shape {4, 16} with int32_t. Inner dim size is 16 * sizeof(int32_t) = 64
  // bytes. This is typically aligned with EIGEN_MAX_ALIGN_BYTES.
  HostBuffer host_buffer = AsHostBuffer<int32_t>(
      {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
       33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
       49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64},
      xla::ifrt::Shape({4, 16}));
  std::vector<HostBuffer> output;
  bool easy_cases_done = false;
  ASSERT_OK(SplitEasyCases(&host_buffer, {1, 2, 1}, output, easy_cases_done));
  EXPECT_TRUE(easy_cases_done);

  HostBuffer expected1 = AsHostBuffer<int32_t>(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
      xla::ifrt::Shape({1, 16}));
  HostBuffer expected2 = AsHostBuffer<int32_t>(
      {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
       33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48},
      xla::ifrt::Shape({2, 16}));
  HostBuffer expected3 = AsHostBuffer<int32_t>(
      {49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64},
      xla::ifrt::Shape({1, 16}));

  EXPECT_THAT(output,
              ElementsAre(HostBufferEq(expected1), HostBufferEq(expected2),
                          HostBufferEq(expected3)));
  EXPECT_EQ(output[0].host_array()->raw_data(),
            host_buffer.host_array()->raw_data());
  EXPECT_EQ(output[1].host_array()->raw_data(),
            static_cast<const int32_t*>(host_buffer.host_array()->raw_data()) +
                16);
  EXPECT_EQ(output[2].host_array()->raw_data(),
            static_cast<const int32_t*>(host_buffer.host_array()->raw_data()) +
                (16 * 3));
}

TEST_F(HostBufferSplitEasyCasesTest, SplitEasyCasesAlignedForString) {
  // Shape {4, 8} with each element being a string of length 4.
  tsl::tstring str_data[32] = {
      "aaaa", "bbbb",  "cccc",  "dddd", "eeee", "ffff",  "gggg", "hhhh",
      "iiii", "jjjjj", "kkkkk", "llll", "mmmm", "nnnn",  "oooo", "pppp",
      "qqqq", "rrrr",  "ssss",  "tttt", "uuuu", "vvvvv", "wwww", "xxxx",
      "yyyy", "zzzz",  "aaaa",  "bbbb", "cccc", "dddd",  "eeee", "ffff",
  };
  HostBuffer host_buffer(str_data,
                         xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                         xla::ifrt::Shape({4, 8}));

  std::vector<int64_t> split_sizes = {1, 2, 1};
  std::vector<HostBuffer> output;
  bool easy_cases_done = false;
  ASSERT_OK(SplitEasyCases(&host_buffer, split_sizes, output, easy_cases_done));

  EXPECT_TRUE(easy_cases_done);
  tsl::tstring expected_str_1[8] = {
      "aaaa", "bbbb", "cccc", "dddd", "eeee", "ffff", "gggg", "hhhh",
  };
  HostBuffer expected_output_1(
      expected_str_1, xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
      xla::ifrt::Shape({1, 8}));
  tsl::tstring expected_str_2[16] = {
      "iiii", "jjjjj", "kkkkk", "llll", "mmmm", "nnnn",  "oooo", "pppp",
      "qqqq", "rrrr",  "ssss",  "tttt", "uuuu", "vvvvv", "wwww", "xxxx",
  };
  HostBuffer expected_output_2(
      expected_str_2, xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
      xla::ifrt::Shape({2, 8}));
  tsl::tstring expected_str_3[8] = {
      "yyyy", "zzzz", "aaaa", "bbbb", "cccc", "dddd", "eeee", "ffff",
  };
  HostBuffer expected_output_3(
      expected_str_3, xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
      xla::ifrt::Shape({1, 8}));
  EXPECT_THAT(output, ElementsAre(HostBufferEq(expected_output_1),
                                  HostBufferEq(expected_output_2),
                                  HostBufferEq(expected_output_3)));
}

TEST_F(HostBufferSplitEasyCasesTest, SplitEasyCasesAlignedSubByteTypesU4) {
  // Shape {4, EIGEN_MAX_ALIGN_BYTES} with kU4.
  // Inner dim size is EIGEN_MAX_ALIGN_BYTES which should always be aligned.
  std::vector<int> u4_data;
  for (int i = 0; i < 128; ++i) {
    u4_data.push_back(i % 16);
  }

  // Create a host buffer with second row EIGEN_MAX_ALIGN_BYTES to ensure
  // sub-byte types are handled correctly.
  HostBuffer host_buffer = AsHostBuffer<tsl::uint4, int>(
      u4_data, xla::ifrt::Shape({4, EIGEN_MAX_ALIGN_BYTES}));

  std::vector<HostBuffer> output;
  bool easy_cases_done = false;
  ASSERT_OK(SplitEasyCases(&host_buffer, {1, 2, 1}, output, easy_cases_done));
  EXPECT_TRUE(easy_cases_done);

  // Expected splits are created from slices of the original u2_data.
  HostBuffer expected1 = AsHostBuffer<tsl::uint4, int>(
      std::vector<int>(u4_data.begin(),
                       u4_data.begin() + EIGEN_MAX_ALIGN_BYTES),
      xla::ifrt::Shape({1, EIGEN_MAX_ALIGN_BYTES}));
  HostBuffer expected2 = AsHostBuffer<tsl::uint4, int>(
      std::vector<int>(u4_data.begin() + EIGEN_MAX_ALIGN_BYTES,
                       u4_data.begin() + (EIGEN_MAX_ALIGN_BYTES * 3)),
      xla::ifrt::Shape({2, EIGEN_MAX_ALIGN_BYTES}));
  HostBuffer expected3 = AsHostBuffer<tsl::uint4, int>(
      std::vector<int>(u4_data.begin() + (EIGEN_MAX_ALIGN_BYTES * 3),
                       u4_data.begin() + (EIGEN_MAX_ALIGN_BYTES * 4)),
      xla::ifrt::Shape({1, EIGEN_MAX_ALIGN_BYTES}));

  EXPECT_THAT(output,
              ElementsAre(HostBufferEq(expected1), HostBufferEq(expected2),
                          HostBufferEq(expected3)));
  EXPECT_EQ(output[0].host_array()->raw_data(),
            host_buffer.host_array()->raw_data());
  EXPECT_EQ(output[1].host_array()->raw_data(),
            static_cast<const char*>(host_buffer.host_array()->raw_data()) +
                EIGEN_MAX_ALIGN_BYTES);
  EXPECT_EQ(output[2].host_array()->raw_data(),
            static_cast<const char*>(host_buffer.host_array()->raw_data()) +
                (EIGEN_MAX_ALIGN_BYTES * 3));
}

TEST_F(HostBufferSplitEasyCasesTest, SplitEasyCasesNotEasy) {
  // Shape {3, 3} with int32_t. Inner dim size is 3 * 4 = 12 bytes.
  // This is not aligned to EIGEN_MAX_ALIGN_BYTES (typically 16 or 32).
  HostBuffer host_buffer = AsHostBuffer<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                 xla::ifrt::Shape({3, 3}));
  std::vector<HostBuffer> output;
  bool easy_cases_done = false;
  ASSERT_OK(SplitEasyCases(&host_buffer, {1, 2}, output, easy_cases_done));
  EXPECT_FALSE(easy_cases_done);
  EXPECT_TRUE(output.empty());
}

TEST(HostBufferProtoTest, ToAndFromProtoNumeric) {
  HostBuffer original_buffer =
      AsHostBuffer<int32_t>({1, 2, 3, 4, 5, 6, 7, 8}, xla::ifrt::Shape({2, 4}));

  HostBufferProto proto =
      HostBuffer::ToHostBufferProto(std::move(original_buffer));

  // Verify the contents of the intermediate proto.
  EXPECT_EQ(proto.dtype().kind(), xla::ifrt::DType::Kind::kS32);
  EXPECT_THAT(proto.shape().dims(), ElementsAre(2, 4));
  std::vector<int32_t> expected_data = {1, 2, 3, 4, 5, 6, 7, 8};
  absl::string_view expected_data_bytes(
      reinterpret_cast<const char*>(expected_data.data()),
      expected_data.size() * sizeof(int32_t));
  EXPECT_EQ(proto.raw_data(), expected_data_bytes);

  TF_ASSERT_OK_AND_ASSIGN(HostBuffer new_buffer, HostBuffer::FromProto(proto));

  HostBuffer expected_buffer =
      AsHostBuffer<int32_t>({1, 2, 3, 4, 5, 6, 7, 8}, xla::ifrt::Shape({2, 4}));

  EXPECT_THAT(new_buffer, HostBufferEq(expected_buffer));
  EXPECT_EQ(new_buffer.dtype(), expected_buffer.dtype());
  EXPECT_EQ(new_buffer.shape(), expected_buffer.shape());
}

TEST(HostBufferProtoTest, ToAndFromProtoString) {
  tsl::tstring str_data[4] = {"hello", "world", "test", "string"};
  HostBuffer original_buffer(str_data,
                             xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                             xla::ifrt::Shape({2, 2}));

  HostBufferProto proto =
      HostBuffer::ToHostBufferProto(std::move(original_buffer));

  // Verify the contents of the intermediate proto.
  EXPECT_EQ(proto.dtype().kind(), xla::ifrt::DType::Kind::kString);
  EXPECT_THAT(proto.shape().dims(), ElementsAre(2, 2));
  EXPECT_THAT(proto.string_data().elements(),
              ElementsAre("hello", "world", "test", "string"));

  TF_ASSERT_OK_AND_ASSIGN(HostBuffer new_buffer, HostBuffer::FromProto(proto));

  HostBuffer expected_buffer(str_data,
                             xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                             xla::ifrt::Shape({2, 2}));

  EXPECT_THAT(new_buffer, HostBufferEq(expected_buffer));
  EXPECT_EQ(new_buffer.dtype(), expected_buffer.dtype());
  EXPECT_EQ(new_buffer.shape(), expected_buffer.shape());
}

TEST(HostBufferProtoTest, FromProtoNonEmptyShapeWithEmptyDataReturnsError) {
  HostBufferProto proto;
  *proto.mutable_dtype() =
      xla::ifrt::DType(xla::ifrt::DType::Kind::kS32).ToProto();
  proto.mutable_shape()->add_dims(2);
  proto.mutable_shape()->add_dims(2);

  // No raw_data or string_data is set.
  EXPECT_THAT(HostBuffer::FromProto(proto),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace jsv
