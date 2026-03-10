#include "runtime/interpreter/orbax_type_utils.h"  // from @neptune

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "third_party/py/orbax/experimental/model/core/protos/type.pb.h"
#include "xla/xla_data.pb.h"  // from @xla

namespace jsv {

absl::StatusOr<xla::PrimitiveType> ToPrimitiveType(
    orbax_model_type::DType type) {
  switch (type) {
    case orbax_model_type::DType::i1:
      return xla::PrimitiveType::PRED;
    case orbax_model_type::DType::f8_e5m2:
      return xla::PrimitiveType::F8E5M2;
    case orbax_model_type::DType::f8_e4m3fn:
      return xla::PrimitiveType::F8E4M3FN;
    case orbax_model_type::DType::bf16:
      return xla::PrimitiveType::BF16;
    case orbax_model_type::DType::f16:
      return xla::PrimitiveType::F16;
    case orbax_model_type::DType::f32:
      return xla::PrimitiveType::F32;
    case orbax_model_type::DType::f64:
      return xla::PrimitiveType::F64;
    case orbax_model_type::DType::c64:
      return xla::PrimitiveType::C64;
    case orbax_model_type::DType::si4:
      return xla::PrimitiveType::S4;
    case orbax_model_type::DType::si8:
      return xla::PrimitiveType::S8;
    case orbax_model_type::DType::si16:
      return xla::PrimitiveType::S16;
    case orbax_model_type::DType::si32:
      return xla::PrimitiveType::S32;
    case orbax_model_type::DType::si64:
      return xla::PrimitiveType::S64;
    case orbax_model_type::DType::ui4:
      return xla::PrimitiveType::U4;
    case orbax_model_type::DType::ui8:
      return xla::PrimitiveType::U8;
    case orbax_model_type::DType::ui16:
      return xla::PrimitiveType::U16;
    case orbax_model_type::DType::ui32:
      return xla::PrimitiveType::U32;
    case orbax_model_type::DType::ui64:
      return xla::PrimitiveType::U64;
    case orbax_model_type::DType::c128:
      return xla::PrimitiveType::C128;
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported type ", type));
  }
}

}  // namespace jsv
