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

#include "runtime/interpreter/orbax_type_utils.h"  // from @xla_serving

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "protos/type.pb.h"  // from @xla_serving
#include "xla/xla_data.pb.h"  // from @xla

namespace xla_serving {

absl::StatusOr<xla::PrimitiveType> ToPrimitiveType(
    neptune_model_type::DType type) {
  switch (type) {
    case neptune_model_type::DType::i1:
      return xla::PrimitiveType::PRED;
    case neptune_model_type::DType::f8_e5m2:
      return xla::PrimitiveType::F8E5M2;
    case neptune_model_type::DType::f8_e4m3fn:
      return xla::PrimitiveType::F8E4M3FN;
    case neptune_model_type::DType::bf16:
      return xla::PrimitiveType::BF16;
    case neptune_model_type::DType::f16:
      return xla::PrimitiveType::F16;
    case neptune_model_type::DType::f32:
      return xla::PrimitiveType::F32;
    case neptune_model_type::DType::f64:
      return xla::PrimitiveType::F64;
    case neptune_model_type::DType::c64:
      return xla::PrimitiveType::C64;
    case neptune_model_type::DType::si4:
      return xla::PrimitiveType::S4;
    case neptune_model_type::DType::si8:
      return xla::PrimitiveType::S8;
    case neptune_model_type::DType::si16:
      return xla::PrimitiveType::S16;
    case neptune_model_type::DType::si32:
      return xla::PrimitiveType::S32;
    case neptune_model_type::DType::si64:
      return xla::PrimitiveType::S64;
    case neptune_model_type::DType::ui4:
      return xla::PrimitiveType::U4;
    case neptune_model_type::DType::ui8:
      return xla::PrimitiveType::U8;
    case neptune_model_type::DType::ui16:
      return xla::PrimitiveType::U16;
    case neptune_model_type::DType::ui32:
      return xla::PrimitiveType::U32;
    case neptune_model_type::DType::ui64:
      return xla::PrimitiveType::U64;
    case neptune_model_type::DType::c128:
      return xla::PrimitiveType::C128;
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported type ", type));
  }
}

}  // namespace xla_serving
