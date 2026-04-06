#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_ORBAX_TYPE_UTILS_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_ORBAX_TYPE_UTILS_H_

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "protos/type.pb.h"  // from @neptune
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla

namespace neptune {

// Converts an `neptune_model_type::DType` to `xla::PrimitiveType`.
absl::StatusOr<xla::PrimitiveType> ToPrimitiveType(
    neptune_model_type::DType type);

// Converts an `neptune_model_type::DType` to `xla::ifrt::DType`.
absl::StatusOr<xla::ifrt::DType> ToIfrtDType(neptune_model_type::DType type);

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_ORBAX_TYPE_UTILS_H_
