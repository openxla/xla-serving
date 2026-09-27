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

#include "runtime/core/host_buffer_utils.h"  // from @xla_serving

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/strings/escaping.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "Eigen/Core"
#include "xla/overflow_util.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "tsl/platform/bfloat16.h"  // from @tsl
#include "tsl/platform/tstring.h"  // from @tsl

namespace xla_serving {
namespace {

std::optional<int64_t> GetByteSize(xla::ifrt::DType dtype) {
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
  _M(Eigen::half, kF16)              \
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
void AppendRangeTyped<bool>(std::string* out, const void* data,
                            int64_t byte_stride, int64_t count) {
  for (int64_t i = 0; i < count; ++i) {
    if (i != 0) absl::StrAppend(out, ", ");
    // Read through `char` rather than `bool`. Loading a `bool` whose value
    // representation is neither 0 nor 1 is undefined behaviour, and the buffer
    // is caller-supplied memory that we cannot assume is normalized.
    absl::StrAppend(out, *reinterpret_cast<const char*>(data) != 0 ? 1 : 0);
    data = reinterpret_cast<const char*>(data) + byte_stride;
  }
}

template <>
void AppendRangeTyped<Eigen::half>(std::string* out, const void* data,
                                   int64_t byte_stride, int64_t count) {
  for (int64_t i = 0; i < count; ++i) {
    if (i != 0) absl::StrAppend(out, ", ");
    absl::StrAppend(
        out, static_cast<float>(*reinterpret_cast<const Eigen::half*>(data)));
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
                 int64_t byte_stride, int64_t count, int64_t max_entries = -1) {
  if (max_entries == 0 && count > 0) {
    absl::StrAppend(out, "...");
    return;
  } else if (max_entries >= 0 && count > 2 * max_entries) {
    AppendRange(out, dtype, data, byte_stride, max_entries, -1);
    absl::StrAppend(out, ", ..., ");
    const char* end_data = reinterpret_cast<const char*>(data) +
                           (count - max_entries) * byte_stride;
    AppendRange(out, dtype, end_data, byte_stride, max_entries, -1);
    return;
  }
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
  std::optional<int64_t> stride = GetByteSize(dtype);
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
                     absl::Span<int64_t const> counts, int64_t max_entries) {
  if (byte_strides.empty()) {
    absl::StrAppend(out, "(");
    AppendRange(out, dtype, data, 0, 1);
    absl::StrAppend(out, ")");
  } else if (byte_strides.size() == 1) {
    absl::StrAppend(out, "[");
    AppendRange(out, dtype, data, byte_strides[0], counts[0], max_entries);
    absl::StrAppend(out, "]");
  } else {
    absl::StrAppend(out, "[\n");
    for (size_t i = 0; i < counts[0]; ++i) {
      if (max_entries >= 0 && counts[0] > 2 * max_entries && i == max_entries) {
        AppendIndent(out, indent + 1);
        absl::StrAppend(out, "...");
        int64_t skipped = counts[0] - 2 * max_entries;
        data = reinterpret_cast<const char*>(data) + skipped * byte_strides[0];
        i = counts[0] - max_entries;
        if (max_entries == 0) {
          // The ellipsis stands for every entry, so no trailing group follows.
          absl::StrAppend(out, "\n");
          break;
        }
        absl::StrAppend(out, ",\n");
      }
      AppendIndent(out, indent + 1);
      AppendRecursive(out, dtype, data, indent + 1, byte_strides.subspan(1),
                      counts.subspan(1), max_entries);
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

std::string SummarizeBuffer(
    xla::ifrt::DType dtype, absl::Span<int64_t const> shape, const void* data,
    int64_t max_entries,
    std::optional<absl::Span<int64_t const>> byte_strides) {
  // Any error is placed in the returned string.
  if (!GetByteSize(dtype).has_value()) {
    return absl::StrCat("Wrong dtype for Buffer: ", dtype);
  }
  // Put an error string for null data unless the number of elements is zero.
  if (data == nullptr) {
    int64_t num_elements = 1;
    for (int64_t dim : shape) {
      if (dim <= 0) {
        num_elements = 0;
        break;
      }
      auto [next_num_elements, overflow] =
          xla::OverflowSafeMultiply(num_elements, dim);
      if (overflow) {
        return absl::StrCat(
            "Overflowing dimensions in uninitialized Buffer of type ", dtype,
            ": ", xla::ifrt::Shape(shape));
      }
      num_elements = next_num_elements;
    }
    if (num_elements > 0) {
      return absl::StrCat("Uninitialized Buffer of ", num_elements,
                          " elements of type ", dtype);
    }
  }
  std::string ret;
  if (byte_strides.has_value()) {
    AppendRecursive(&ret, dtype, data, 0, *byte_strides, shape, max_entries);
    return ret;
  }
  std::optional<absl::InlinedVector<int64_t, 4>> default_strides =
      CreateDefaultStrides(shape, dtype);
  if (!default_strides.has_value()) {
    return absl::StrCat("Failed to create default strides for Buffer of type ",
                        dtype);
  }
  AppendRecursive(&ret, dtype, data, 0, *default_strides, shape, max_entries);
  return ret;
}

}  // namespace xla_serving
