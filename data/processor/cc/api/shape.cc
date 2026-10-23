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

#include "data/processor/cc/api/shape.h"  // from @xla_serving

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "xla/python/ifrt/shape.h"  // from @xla

namespace xla_serving::data {
namespace composite {
namespace api {
namespace {

// "[?,4]", or "[]" for a scalar.
std::string DimsDebugString(absl::Span<const Dim> dims) {
  return absl::StrCat("[", absl::StrJoin(dims, ","), "]");
}

// Returns OK iff the product of the known, nonzero sizes in `dims` fits in
// int64_t: the PartialShape invariant. Skipping the zeros, as TF's
// BuildTensorShapeBase does, makes the product fit in any order.
absl::Status CheckNumElements(absl::Span<const Dim> dims) {
  int64_t product = 1;
  for (const Dim& d : dims) {
    const std::optional<int64_t> size = d.Value();
    if (!size.has_value() || *size <= 0) continue;
    if (__builtin_mul_overflow(product, *size, &product)) [[unlikely]] {
      return absl::InvalidArgumentError(
          absl::StrCat("Shape ", DimsDebugString(dims),
                       " is too large (more than 2**63 - 1 entries)"));
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<Dim> Dim::FromSize(int64_t size) {
  if (size < 0) [[unlikely]] {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected a non-negative size, got ", size));
  }
  return Dim(size);
}

std::string Dim::DebugString() const {
  if (!IsKnown()) return "?";
  return absl::StrCat(rep_);
}

// From InferenceContext in tensorflow/core/framework/shape_inference.cc.

absl::StatusOr<Dim> Dim::Merge(const Dim& a, const Dim& b) {
  if (!b.IsKnown()) return a;
  if (!a.IsKnown()) return b;
  if (a == b) return a;
  return absl::InvalidArgumentError(absl::StrCat(
      "Dimensions must be equal, but are ", *a.Value(), " and ", *b.Value()));
}

Dim Dim::Relax(const Dim& a, const Dim& b) {
  if (a.IsKnown() && a == b) return a;
  return Dim::Unknown();
}

absl::StatusOr<Dim> Dim::Add(const Dim& a, const Dim& b) {
  if (a.Value() == 0) return b;
  if (b.Value() == 0) return a;
  if (!a.IsKnown() || !b.IsKnown()) return Dim::Unknown();
  const int64_t x = *a.Value();
  const int64_t y = *b.Value();
  int64_t sum;
  if (__builtin_add_overflow(x, y, &sum)) [[unlikely]] {
    return absl::InvalidArgumentError(
        absl::StrCat("Dimension size overflow from adding ", x, " and ", y));
  }
  return Dim(sum);
}

absl::StatusOr<Dim> Dim::Subtract(const Dim& a, const Dim& b) {
  if (b.Value() == 0) return a;
  if (!a.IsKnown() || !b.IsKnown()) return Dim::Unknown();
  const int64_t x = *a.Value();
  const int64_t y = *b.Value();
  if (x < y) [[unlikely]] {
    return absl::InvalidArgumentError(absl::StrCat(
        "Negative dimension size caused by subtracting ", y, " from ", x));
  }
  return Dim(x - y);
}

absl::StatusOr<Dim> Dim::Multiply(const Dim& a, const Dim& b) {
  if (a.Value() == 0) return a;
  if (b.Value() == 0) return b;
  if (a.Value() == 1) return b;
  if (b.Value() == 1) return a;
  if (!a.IsKnown() || !b.IsKnown()) return Dim::Unknown();
  const int64_t x = *a.Value();
  const int64_t y = *b.Value();
  int64_t product;
  if (__builtin_mul_overflow(x, y, &product)) [[unlikely]] {
    return absl::InvalidArgumentError(absl::StrCat(
        "Dimension size overflow from multiplying ", x, " and ", y));
  }
  return Dim(product);
}

absl::StatusOr<Dim> Dim::Divide(const Dim& a, int64_t divisor,
                                bool evenly_divisible) {
  if (divisor <= 0) [[unlikely]] {
    return absl::InvalidArgumentError(
        absl::StrCat("Divisor must be positive but is ", divisor));
  }
  if (divisor == 1) return a;
  if (!a.IsKnown()) return Dim::Unknown();
  const int64_t x = *a.Value();
  if (evenly_divisible && x % divisor != 0) [[unlikely]] {
    return absl::InvalidArgumentError(absl::StrCat(
        "Dimension size must be evenly divisible by ", divisor, " but is ", x));
  }
  return Dim(x / divisor);
}

Dim Dim::Min(const Dim& a, const Dim& b) {
  if (a.Value() == 0) return a;
  if (b.Value() == 0) return b;
  if (!a.IsKnown() || !b.IsKnown()) return Dim::Unknown();
  return *a.Value() <= *b.Value() ? a : b;
}

Dim Dim::Max(const Dim& a, const Dim& b) {
  if (!a.IsKnown() || !b.IsKnown()) return Dim::Unknown();
  return *a.Value() >= *b.Value() ? a : b;
}

Dim PartialShape::num_elements() const {
  bool fully_known = true;
  for (const Dim& d : dims_) {
    if (d.Value() == 0) return Dim::Static(0);
    if (!d.IsKnown()) fully_known = false;
  }
  if (!fully_known) return Dim::Unknown();
  int64_t product = 1;
  for (const Dim& d : dims_) {
    if (__builtin_mul_overflow(product, *d.Value(), &product)) [[unlikely]] {
      // Unreachable: Validation in Create ensures the product fits in int64_t.
      return Dim::Unknown();
    }
  }
  return Dim(product);
}

std::string PartialShape::DebugString() const { return DimsDebugString(dims_); }

PartialShape::PartialShape(const xla::ifrt::Shape& shape) {
  dims_.reserve(shape.dims().size());
  for (const int64_t size : shape.dims()) {
    DCHECK_GE(size, 0);
    dims_.push_back(Dim(size));
  }
  DCHECK_OK(CheckNumElements(dims_));
}

bool PartialShape::IsFullyKnown() const {
  return absl::c_all_of(dims_, [](const Dim& d) { return d.IsKnown(); });
}

PartialShapeProto PartialShape::ToProto() const {
  PartialShapeProto proto;
  for (const Dim& d : dims_) {
    DimProto* dim = proto.add_dims();
    if (const std::optional<int64_t> size = d.Value(); size.has_value()) {
      dim->set_size(*size);
    }
  }
  return proto;
}

absl::StatusOr<PartialShape> PartialShape::Create(absl::Span<const Dim> dims) {
  if (absl::Status status = CheckNumElements(dims); !status.ok()) [[unlikely]] {
    return status;
  }
  return PartialShape(Dimensions(dims.begin(), dims.end()));
}

absl::StatusOr<PartialShape> PartialShape::Create(
    std::initializer_list<absl::StatusOr<Dim>> dims) {
  Dimensions result;
  result.reserve(dims.size());
  for (const absl::StatusOr<Dim>& d : dims) {
    if (!d.ok()) [[unlikely]] {
      return absl::Status(d.status().code(),
                          absl::StrCat("Dimension ", result.size(), ": ",
                                       d.status().message()));
    }
    result.push_back(*d);
  }
  return Create(absl::MakeConstSpan(result));
}

absl::StatusOr<PartialShape> PartialShape::FromProto(
    const PartialShapeProto& proto) {
  Dimensions dims;
  dims.reserve(proto.dims_size());
  for (int i = 0; i < proto.dims_size(); ++i) {
    const DimProto& d = proto.dims(i);
    if (!d.has_size()) {
      dims.push_back(Dim::Unknown());
      continue;
    }
    if (d.size() < 0) [[unlikely]] {
      return absl::InvalidArgumentError(
          absl::StrCat("Dimension ", i, " has a negative size ", d.size(),
                       "; leave `size` unset for an unknown dimension"));
    }
    dims.push_back(Dim(d.size()));
  }
  return Create(dims);
}

absl::StatusOr<PartialShape> PartialShape::MergeWith(
    const PartialShape& shape) const {
  if (dims_.size() != shape.dims_.size()) [[unlikely]] {
    return absl::InvalidArgumentError(absl::StrCat(
        "PartialShape: Incompatible ranks during merge: ", dims_.size(),
        " vs. ", shape.dims_.size()));
  }
  Dimensions result;
  result.reserve(dims_.size());
  for (size_t i = 0; i < dims_.size(); ++i) {
    // Merge fails iff both sizes are known and differ.
    absl::StatusOr<Dim> merged = Dim::Merge(dims_[i], shape.dims_[i]);
    if (!merged.ok()) [[unlikely]] {
      return absl::InvalidArgumentError(absl::StrCat(
          "PartialShape: Incompatible shapes during merge: ", DebugString(),
          " vs. ", shape.DebugString()));
    }
    result.push_back(*merged);
  }
  // Each side satisfies the invariant, but the merged shape may not: [2**40,?]
  // and [?,2**40] merge to [2**40,2**40]. Create rejects it.
  return Create(result);
}

absl::StatusOr<PartialShape> PartialShape::RelaxWith(
    const PartialShape& shape) const {
  if (dims_.size() != shape.dims_.size()) [[unlikely]] {
    return absl::InvalidArgumentError(
        absl::StrCat("Cannot relax shapes of different ranks: ", DebugString(),
                     " vs. ", shape.DebugString()));
  }
  Dimensions result;
  result.reserve(dims_.size());
  for (size_t i = 0; i < dims_.size(); ++i) {
    result.push_back(Dim::Relax(dims_[i], shape.dims_[i]));
  }
  // The known sizes of the result are known sizes of `*this`, so the result
  // satisfies the invariant and Create succeeds.
  return Create(result);
}

absl::StatusOr<PartialShape> PartialShape::BroadcastWith(
    const PartialShape& shape) const {
  const size_t rank = std::max(dims_.size(), shape.dims_.size());
  const size_t offset = rank - dims_.size();
  const size_t shape_offset = rank - shape.dims_.size();
  Dimensions result;
  result.reserve(rank);
  for (size_t i = 0; i < rank; ++i) {
    const Dim a = i < offset ? Dim::Static(1) : dims_[i - offset];
    const Dim b =
        i < shape_offset ? Dim::Static(1) : shape.dims_[i - shape_offset];
    if (a.Value() == 1) {
      result.push_back(b);
    } else if (b.Value() == 1) {
      result.push_back(a);
    } else if (a.IsKnown() && b.IsKnown()) {
      if (!(a == b)) [[unlikely]] {
        return absl::InvalidArgumentError(
            absl::StrCat("Incompatible shapes: ", DebugString(), " vs. ",
                         shape.DebugString()));
      }
      result.push_back(a);
    } else {
      // A known size wins; two unknown dimensions stay unknown.
      result.push_back(a.IsKnown() ? a : b);
    }
  }
  // [2**40,1] and [1,2**40] broadcast to [2**40,2**40]. Create rejects it.
  return Create(result);
}

bool PartialShape::IsIdenticalTo(const PartialShape& shape) const {
  return dims_ == shape.dims_;
}

bool PartialShape::IsCompatibleWith(const PartialShape& shape) const {
  if (dims_.size() != shape.dims_.size()) return false;
  for (size_t i = 0; i < dims_.size(); ++i) {
    const Dim& a = dims_[i];
    const Dim& b = shape.dims_[i];
    if (a.IsKnown() && b.IsKnown() && !(a == b)) return false;
  }
  return true;
}

absl::StatusOr<xla::ifrt::Shape> PartialShape::GetStatic() const {
  xla::ifrt::Shape::Dimensions sizes;
  sizes.reserve(dims_.size());
  for (size_t i = 0; i < dims_.size(); ++i) {
    const std::optional<int64_t> size = dims_[i].Value();
    if (!size.has_value()) [[unlikely]] {
      return absl::InvalidArgumentError(
          absl::StrCat("Shape ", DebugString(),
                       " is not fully known: dimension ", i, " is unknown"));
    }
    sizes.push_back(*size);
  }
  return xla::ifrt::Shape(sizes);
}

PartialValue PartialValue::Known(absl::Span<const int64_t> values) {
  return PartialValue(
      std::vector<std::optional<int64_t>>(values.begin(), values.end()));
}

PartialValue PartialValue::FromElements(
    std::vector<std::optional<int64_t>> elements) {
  return PartialValue(std::move(elements));
}

PartialValue PartialValue::FromShape(const PartialShape& shape) {
  std::vector<std::optional<int64_t>> elements;
  elements.reserve(shape.dims().size());
  for (const Dim& d : shape.dims()) {
    elements.push_back(d.Value());
  }
  return PartialValue(std::move(elements));
}

std::optional<int64_t> PartialValue::element(int64_t i) const {
  DCHECK_GE(i, 0);
  DCHECK_LT(i, size());
  return elements_[i];
}

bool PartialValue::IsFullyKnown() const {
  return absl::c_all_of(
      elements_, [](const std::optional<int64_t>& e) { return e.has_value(); });
}

std::optional<std::vector<int64_t>> PartialValue::AsInts() const {
  std::vector<int64_t> values;
  values.reserve(elements_.size());
  for (const std::optional<int64_t>& e : elements_) {
    if (!e.has_value()) return std::nullopt;
    values.push_back(*e);
  }
  return values;
}

absl::StatusOr<PartialShape> PartialValue::AsShape() const {
  PartialShape::Dimensions dims;
  dims.reserve(elements_.size());
  for (int64_t i = 0; i < size(); ++i) {
    const std::optional<int64_t>& e = elements_[i];
    Dim dim = Dim::Unknown();
    if (e.has_value()) {
      absl::StatusOr<Dim> known = Dim::FromSize(*e);
      if (!known.ok()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Expected a non-negative size for dimension ", i, ", got ", *e,
            "; a literal -1 is not an unknown dimension"));
      }
      dim = *known;
    }
    dims.push_back(dim);
  }
  return PartialShape::Create(dims);
}

std::string PartialValue::DebugString() const {
  return absl::StrCat(
      "{",
      absl::StrJoin(elements_, ",",
                    [](std::string* out, const std::optional<int64_t>& e) {
                      if (e.has_value()) {
                        absl::StrAppend(out, *e);
                      } else {
                        absl::StrAppend(out, "?");
                      }
                    }),
      "}");
}

std::ostream& operator<<(std::ostream& os, const Dim& d) {
  return os << d.DebugString();
}

std::ostream& operator<<(std::ostream& os, const PartialShape& shape) {
  return os << shape.DebugString();
}

std::ostream& operator<<(std::ostream& os, const PartialValue& value) {
  return os << value.DebugString();
}

}  // namespace api
}  // namespace composite
}  // namespace xla_serving::data
