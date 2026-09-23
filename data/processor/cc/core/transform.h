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

#ifndef XLA_SERVING_DATA_PROCESSOR_CC_CORE_TRANSFORM_H_
#define XLA_SERVING_DATA_PROCESSOR_CC_CORE_TRANSFORM_H_

#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/transform_context.h"  // from @xla_serving
#include "data/processor/cc/api/type_info.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "data/resource_manager/cc/resource.h"  // from @xla_serving

namespace xla_serving::data {
namespace composite {
namespace core {

// Forward declarations.
class TransformRegistry;
class DynamicDispatchTransform;

}  // namespace core


// TransformBase is the type-erased base class for all transforms.
//
// A transform is an operation that takes a list of values as input
// and produces a list of values as output.
//
// NOTE: Classes deriving from TransformBase that want to be registered
// via ND_REGISTER_TRANSFORM MUST implement:
//
// 1. A static Create method to act as a factory:
//    static absl::StatusOr<std::unique_ptr<TransformBase>> Create(...);
//
// 2. A const operator() to execute the transform with typed arguments
//
// See data/processor/python/examples/ for examples.
class TransformBase {
 public:
  virtual ~TransformBase() = default;
  // Executes the transform.
  //
  // The transform should populate `context.outputs` with the results of the
  // transform. The size of `context.outputs` must match the size of
  // `OutputTypes()`.
  // The caller of Execute must ensure that the types and size of
  // `context.inputs` matches those declared by `InputTypes()`. Likewise, the
  // caller must clear and reserve the appropriate capacity for
  // `context.outputs` based on `OutputTypes()`.
  virtual absl::Status Execute(api::ExecuteContext& context);

  // Propagates shape and dtype signature from inputs to outputs.
  // Users should not implement this directly and should only implement
  // `PropagateSignatureImpl`.
  virtual absl::Status PropagateSignature(
      api::ShapeInferenceContext& context) const;

  // Returns the types of the inputs to the transform.
  virtual absl::Span<const api::TypeInfo> InputTypes() const;
  // Returns the types of the outputs of the transform.
  virtual absl::Span<const api::TypeInfo> OutputTypes() const;

  // Serializes the transform to a configuration proto.
  virtual absl::StatusOr<TransformConfig> ToProto() const = 0;

  // Returns the debug string representation of the transform.
  // If the transform was registered with a name, that name is used.
  // Otherwise, the default DebugString() implementation is used.
  // Note: It is HIGHLY RECOMMENDED every user provide their own
  //       implementation of DebugString, as this default implementation
  //       is for backwards compatibility only.
  virtual absl::string_view DebugString() const {
    if (!default_debug_string_.empty()) {
      return default_debug_string_;
    }
    return "TransformBase(unknown)";
  }

  // Returns a debug string representing how this node can be identified in the
  // execution graph.
  virtual std::string DebugName() const { return "TransformBase"; }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const TransformBase& transform) {
    sink.Append(transform.DebugString());
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const TransformBase& transform) {
    return os << transform.DebugString();
  }

  // Returns all resources used by this transform.
  virtual std::vector<std::shared_ptr<xla_serving::data::Resource>>
  GetAllResources() const {
    return {};
  }

  // Creates a transform from a configuration proto.
  static absl::StatusOr<std::unique_ptr<TransformBase>> FromProto(
      const TransformConfig& config,
      absl::Span<const std::shared_ptr<xla_serving::data::Resource>> resources);

 protected:
  virtual absl::Status PropagateSignatureImpl(
      api::ShapeInferenceContext& context) const;

  void SetDefaultDebugString(absl::string_view debug_string) {
    default_debug_string_ = std::string(debug_string);
  }

 private:
  std::string default_debug_string_;

  friend class core::TransformRegistry;
  friend class core::DynamicDispatchTransform;
};

}  // namespace composite

}  // namespace xla_serving::data

#endif  // XLA_SERVING_DATA_PROCESSOR_CC_CORE_TRANSFORM_H_
