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

#ifndef XLA_SERVING_DATA_PROCESSOR_CC_API_TRANSFORM_CONTEXT_H_
#define XLA_SERVING_DATA_PROCESSOR_CC_API_TRANSFORM_CONTEXT_H_

#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/shape.h"  // from @xla_serving
#include "data/processor/cc/api/value.h"  // from @xla_serving
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla

namespace xla_serving::data {
namespace composite {
namespace api {

struct ExecuteMetadata {
  absl::string_view model_name;
  int64_t model_version;
  uint64_t trace_id;
};

// The trace-time description of a HostBuffer.
struct HostBufferSpec {
  xla::ifrt::DType dtype = xla::ifrt::DType(xla::ifrt::DType::Kind::kInvalid);
  // A scalar by default. Dimensions may be unknown; code that needs the static
  // shape calls `shape.GetStatic()`, which rejects them.
  PartialShape shape = xla::ifrt::Shape({});

  // Structural, as PartialShape's `==`.
  friend bool operator==(const HostBufferSpec&,
                         const HostBufferSpec&) = default;
};

// Wrapper around HostBufferSpec to cleanly propagate shape validation. This
// wrapper currently supports args and list args.
class ShapeInferenceValue {
 public:
  using SpecList = std::vector<HostBufferSpec>;

  ShapeInferenceValue() = default;
  /* implicit */ ShapeInferenceValue(HostBufferSpec spec)
      : data_(std::move(spec)) {}
  /* implicit */ ShapeInferenceValue(SpecList list) : data_(std::move(list)) {}

  bool IsSpec() const { return std::holds_alternative<HostBufferSpec>(data_); }
  bool IsSpecList() const { return std::holds_alternative<SpecList>(data_); }

  const HostBufferSpec& spec() const { return std::get<HostBufferSpec>(data_); }
  const SpecList& spec_list() const { return std::get<SpecList>(data_); }

 private:
  std::variant<std::monostate, HostBufferSpec, SpecList> data_;
};

struct ShapeInferenceContext {
  ExecuteMetadata metadata;
  absl::Span<const ShapeInferenceValue> inputs;
  absl::InlinedVector<ShapeInferenceValue, 4> outputs;
};

// The context of the transform execution. The transform takes the
// inputs from the `context.inputs` and put the outputs to the
// `context.outputs`.
struct ExecuteContext {
  // The metadata of the transform execution.
  ExecuteMetadata metadata;
  // The inputs to the transform.
  absl::Span<const Value> inputs;
  // The outputs of the transform.
  // The execution harness will clear and reserve space automatically
  // for the outputs based on the transform's OutputTypes.
  absl::InlinedVector<Value, 4> outputs;
};

}  // namespace api
}  // namespace composite
}  // namespace xla_serving::data

#endif  // XLA_SERVING_DATA_PROCESSOR_CC_API_TRANSFORM_CONTEXT_H_
