#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_ORBAX_TYPE_UTILS_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_ORBAX_TYPE_UTILS_H_

#include "third_party/absl/status/statusor.h"
#include "third_party/py/orbax/experimental/model/core/protos/type.proto.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/dtype.h"
#include "third_party/tensorflow/compiler/xla/xla_data.proto.h"

namespace jsv {

// Converts an `orbax_model_type::DType` to `xla::PrimitiveType`.
absl::StatusOr<xla::PrimitiveType> ToPrimitiveType(
    orbax_model_type::DType type);

// Converts an `orbax_model_type::DType` to `xla::ifrt::DType`.
absl::StatusOr<xla::ifrt::DType> ToIfrtDType(orbax_model_type::DType type);

}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_ORBAX_TYPE_UTILS_H_
