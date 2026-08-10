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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_HOST_BUFFER_MATCHER_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_HOST_BUFFER_MATCHER_H_

#include "gtest/gtest.h"
#include "runtime/core/host_buffer.h"  // from @neptune

namespace neptune {

// Matcher for neptune::HostBuffer instances. Two HostBuffers match iff
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

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_HOST_BUFFER_MATCHER_H_
