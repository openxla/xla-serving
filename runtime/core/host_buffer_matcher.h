#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_HOST_BUFFER_MATCHER_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_HOST_BUFFER_MATCHER_H_

#include "testing/base/public/gunit.h"
#include "runtime/core/host_buffer.h"  // from @neptune

namespace jsv {

// Matcher for jsv::HostBuffer instances. Two HostBuffers match iff
//
//   - their dtypes are equal,
//   - their shapes are equal,
//   - and their contents are equal.
//
// Their contents are matched by ::testing::Pointwise() after calling .flat<T>()
// method.
// Use this like:
//
//   EXPECT_THAT(lhs, HostBufferEq(rhs));
//
class HostBufferEq {
 public:
  explicit HostBufferEq(const HostBuffer& target) : target_(target) {}

  // Matchers depend on implicit casts. Do not make explicit.
  operator ::testing::Matcher<const HostBuffer&>() const;  // NOLINT

 private:
  const HostBuffer& target_;
};

}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_HOST_BUFFER_MATCHER_H_
