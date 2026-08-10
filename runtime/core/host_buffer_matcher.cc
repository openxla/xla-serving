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

#include "runtime/core/host_buffer_matcher.h"  // from @neptune

#include <ostream>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/host_buffer.h"  // from @neptune
#include "runtime/core/host_buffer_type_utils.h"  // from @neptune
#include "xla/python/ifrt/dtype.h"  // from @xla

namespace neptune {
namespace {

template <typename T>
::testing::Matcher<absl::Span<const T>> MakePointwiseMatcher(
    absl::Span<const T> target) {
  return ::testing::MatcherCast<absl::Span<const T>>(
      ::testing::Pointwise(::testing::Eq(), target));
}

template <>
::testing::Matcher<absl::Span<const float>> MakePointwiseMatcher(
    absl::Span<const float> target) {
  return ::testing::MatcherCast<absl::Span<const float>>(
      ::testing::Pointwise(::testing::FloatEq(), target));
}

template <>
::testing::Matcher<absl::Span<const double>> MakePointwiseMatcher(
    absl::Span<const double> target) {
  return ::testing::MatcherCast<absl::Span<const double>>(
      ::testing::Pointwise(::testing::DoubleEq(), target));
}

template <typename T>
bool MatchAndExplainPointwise(absl::Span<const T> value,
                              absl::Span<const T> target,
                              ::testing::MatchResultListener* listener) {
  return MakePointwiseMatcher<T>(target).MatchAndExplain(value, listener);
}

class HostBufferEqMatcherImpl
    : public ::testing::MatcherInterface<const HostBuffer&> {
 public:
  explicit HostBufferEqMatcherImpl(const HostBuffer& target)
      : target_(target) {}

  void DescribeTo(::std::ostream* os) const override {
    *os << "data type is " << target_.dtype() << ", and shape is "
        << target_.shape();
    switch (target_.dtype().kind()) {
#define CASE_TYPE(T, DTYPE_KIND)                           \
  case DTYPE_KIND: {                                       \
    *os << ", and data ";                                  \
    absl::Span<const T> data(target_.unaligned_flat<T>()); \
    MakePointwiseMatcher<T>(data).DescribeTo(os);          \
    break;                                                 \
  }
      APPLY_POD_CASE(CASE_TYPE);
#undef CASE_TYPE
      default: {
        DLOG(FATAL) << "HostBufferEq matcher unsupported dtype: "
                    << target_.dtype();
      }
    }
  }

  void DescribeNegationTo(::std::ostream* os) const override {
    *os << "doesn't match the expected HostBuffer";
  }

  bool MatchAndExplain(
      const HostBuffer& value,
      ::testing::MatchResultListener* listener) const override {
    const bool dtype_compare = value.dtype() == target_.dtype();
    *listener << "whose data type " << value.dtype()
              << (dtype_compare ? " matches " : " doesn't match ")
              << target_.dtype();

    const bool shape_compare = value.shape() == target_.shape();
    *listener << ", whose shape " << value.shape()
              << (shape_compare ? " matches " : " doesn't match ")
              << target_.shape();

    if (!dtype_compare || !shape_compare) {
      return false;
    }

    *listener << ", ";

    bool result;
    switch (target_.dtype().kind()) {
#define CASE_TYPE(T, DTYPE_KIND)                                           \
  case DTYPE_KIND: {                                                       \
    result = MatchAndExplainPointwise<T>(                                  \
        value.unaligned_flat<T>(), target_.unaligned_flat<T>(), listener); \
    break;                                                                 \
  }

      APPLY_POD_CASE(CASE_TYPE);
#undef CASE_TYPE
      default: {
        DLOG(FATAL) << "HostBufferEq matcher unsupported dtype: "
                    << target_.dtype();
        result = false;
      }
    }

    return result;
  }

 private:
  const HostBuffer target_;
};

}  // namespace

HostBufferEq::operator ::testing::Matcher<const HostBuffer&>() const {
  return ::testing::MakeMatcher(new HostBufferEqMatcherImpl(target_));
}

}  // namespace neptune
