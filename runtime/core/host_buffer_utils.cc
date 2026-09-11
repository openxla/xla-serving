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

#include "runtime/core/host_buffer_utils.h"  // from @neptune

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/strings/escaping.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "xla/overflow_util.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "tsl/platform/bfloat16.h"  // from @tsl
#include "tsl/platform/tstring.h"  // from @tsl

namespace xla_serving {
namespace {

std::optional<int64_t> GetByteSizeOrDie(xla::ifrt::DType dtype) {
  if (dtype.kind() == xla::ifrt::DType::Kind::kString) {
    return sizeof(tsl::tstring);
  }
  std::optional<int64_t> bit_size = dtype.bit_size();
  if (!bit_size.has_value()) {
    LOG_FIRST_N(WARNING, 1) << "DType " << dtype << " has no bit size.";
    return std::nullopt;
  }
  // Pad to whole bytes, consistent with numpy and UnpackArrayList.
  return (*bit_size + 7) / 8;
}

#define AUX_FOR_EACH_INTEGER_DTYPE(_M) \
  _M(int8_t, kS8)                      \
  _M(uint8_t, kU8)                     \
  _M(int16_t, kS16)                    \
  _M(uint16_t, kU16)                   \
  _M(int32_t, kS32)                    \
  _M(uint32_t, kU32)                   \
  _M(int64_t, kS64)                    \
  _M(uint64_t, kU64)

#define AUX_FOR_EACH_FLOAT_DTYPE(_M) \
  _M(double, kF64)                   \
  _M(float, kF32)                    \
  _M(tsl::bfloat16, kBF16)

#define AUX_FOR_EACH_DTYPE(_M)   \
  _M(bool, kPred)                \
  AUX_FOR_EACH_INTEGER_DTYPE(_M) \
  AUX_FOR_EACH_FLOAT_DTYPE(_M)

template <typename cpp_type>
void AppendRangeTyped(std::string* out, const void* data, int64_t byte_stride,
                      int64_t count) {
  for (int64_t i = 0; i < count; ++i) {
    if (i != 0) absl::StrAppend(out, ", ");
    absl::StrAppend(out, *reinterpret_cast<const cpp_type*>(data));
    data = reinterpret_cast<const char*>(data) + byte_stride;
  }
}

template <>
void AppendRangeTyped<tsl::bfloat16>(std::string* out, const void* data,
                                int64_t byte_stride, int64_t count) {
  for (int64_t i = 0; i < count; ++i) {
    if (i != 0) absl::StrAppend(out, ", ");
    absl::StrAppend(
        out, static_cast<float>(*reinterpret_cast<const tsl::bfloat16*>(data)));
    data = reinterpret_cast<const char*>(data) + byte_stride;
  }
}

template <>
void AppendRangeTyped<tsl::tstring>(std::string* out, const void* data,
                                    int64_t byte_stride, int64_t count) {
  for (int64_t i = 0; i < count; ++i) {
    if (i != 0) absl::StrAppend(out, ", ");
    absl::StrAppend(out, "'",
                    absl::CEscape(std::string(
                        *reinterpret_cast<const tsl::tstring*>(data))),
                    "'");
    data = reinterpret_cast<const char*>(data) + byte_stride;
  }
}

void AppendRange(std::string* out, xla::ifrt::DType dtype, const void* data,
                 int64_t byte_stride, int64_t count) {
  switch (dtype.kind()) {
#define DEFINE_DTYPE_PRINTER(cpp_type, enum_tag) \
  case xla::ifrt::DType::enum_tag:               \
    return AppendRangeTyped<cpp_type>(out, data, byte_stride, count);
    AUX_FOR_EACH_DTYPE(DEFINE_DTYPE_PRINTER);
#undef DEFINE_DTYPE_PRINTER
    case xla::ifrt::DType::Kind::kString:
      return AppendRangeTyped<tsl::tstring>(out, data, byte_stride, count);
    default:
      absl::StrAppend(out, "?");
  }
}

void AppendIndent(std::string* out, size_t indent) {
  for (; indent--;) {
    absl::StrAppend(out, "  ");
  }
}

}  // namespace

std::optional<absl::InlinedVector<int64_t, 4>> CreateDefaultStrides(
    absl::Span<int64_t const> dims, xla::ifrt::DType dtype) {
  absl::InlinedVector<int64_t, 4> result;
  result.resize(dims.size());
  std::optional<int64_t> stride = GetByteSizeOrDie(dtype);
  if (!stride.has_value()) {
    return std::nullopt;
  }
  for (size_t i = dims.size(); i > 0;) {
    --i;
    result[i] = stride.value();
    auto [next_stride, overflow] = xla::OverflowSafeMultiply(*stride, dims[i]);
    if (overflow) {
      LOG(WARNING) << "Integer overflow in stride calculation";
      return std::nullopt;
    }
    *stride = next_stride;
  }
  return result;
}

void AppendRecursive(std::string* out, xla::ifrt::DType dtype, const void* data,
                     size_t indent, absl::Span<int64_t const> byte_strides,
                     absl::Span<int64_t const> counts) {
  if (byte_strides.empty()) {
    absl::StrAppend(out, "(");
    AppendRange(out, dtype, data, 0, 1);
    absl::StrAppend(out, ")");
  } else if (byte_strides.size() == 1) {
    absl::StrAppend(out, "[");
    AppendRange(out, dtype, data, byte_strides[0], counts[0]);
    absl::StrAppend(out, "]");
  } else {
    absl::StrAppend(out, "[\n");
    for (size_t i = 0; i < counts[0]; ++i) {
      AppendIndent(out, indent + 1);
      AppendRecursive(out, dtype, data, indent + 1, byte_strides.subspan(1),
                      counts.subspan(1));
      data = reinterpret_cast<const char*>(data) + byte_strides[0];
      if (i + 1 != counts[0]) {
        absl::StrAppend(out, ",\n");
      } else {
        absl::StrAppend(out, "\n");
      }
    }
    AppendIndent(out, indent);
    absl::StrAppend(out, "]");
  }
}

}  // namespace xla_serving
