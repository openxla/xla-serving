#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_ORBAX_TYPE_UTILS_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_ORBAX_TYPE_UTILS_H_

#include "absl/status/statusor.h"  // from @com_google_absl
#include "third_party/py/orbax/experimental/model/core/protos/type.pb.h"
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla

namespace jsv {

// Converts an `orbax_model_type::DType` to `xla::PrimitiveType`.
absl::StatusOr<xla::PrimitiveType> ToPrimitiveType(
    orbax_model_type::DType type);

// Converts an `orbax_model_type::DType` to `xla::ifrt::DType`.
absl::StatusOr<xla::ifrt::DType> ToIfrtDType(orbax_model_type::DType type);

}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_ORBAX_TYPE_UTILS_H_
