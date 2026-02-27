#include "third_party/neptune/runtime/core/host_buffer_test_utils.h"

#include <cstdint>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/shape.h"

namespace jsv {
namespace {

using ::testing::HasSubstr;

TEST(AsHostBufferTest, Basic) {
  ASSERT_NO_FATAL_FAILURE(
      AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({4})));
  ASSERT_NO_FATAL_FAILURE(
      AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({2, 2})));
}

TEST(AsHostBufferTest, Scalar) {
  ASSERT_NO_FATAL_FAILURE(AsHostBuffer<int32_t>(1));
}

TEST(AsHostBufferDeathTest, MismatchShape) {
  EXPECT_DEATH(AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({2, 3})),
               HasSubstr("does not match the shape"));
}

}  // namespace
}  // namespace jsv
