#ifndef THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_CC_CORE_TRANSFORM_H_
#define THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_CC_CORE_TRANSFORM_H_

#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/transform_context.h"  // from @neptune
#include "data/processor/cc/api/type_info.h"  // from @neptune
#include "data/processor/proto/transform_config.pb.h"  // from @neptune
#include "data/resource_manager/cc/resource.h"  // from @neptune

namespace neptune::data {
namespace composite {

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
// See third_party/neptune/data/processor/python/examples/ for
// examples.

namespace core {
class TransformRegistry;
}  // namespace core

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
  absl::Status PropagateSignature(api::ShapeInferenceContext& context) const;

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
  virtual std::vector<std::shared_ptr<neptune::data::Resource>>
  GetAllResources() const {
    return {};
  }

  // Creates a transform from a configuration proto.
  static absl::StatusOr<std::unique_ptr<TransformBase>> FromProto(
      const TransformConfig& config,
      absl::Span<const std::shared_ptr<neptune::data::Resource>> resources);

 protected:
  virtual absl::Status PropagateSignatureImpl(
      api::ShapeInferenceContext& context) const;

 private:
  std::string default_debug_string_;

  void SetDefaultDebugString(absl::string_view debug_string) {
    default_debug_string_ = std::string(debug_string);
  }

  friend class core::TransformRegistry;
};

}  // namespace composite
}  // namespace neptune::data

#endif  // THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_CC_CORE_TRANSFORM_H_
