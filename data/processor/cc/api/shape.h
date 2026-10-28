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

// Partially known shapes for trace-time shape inference.
//
// PartialShape mirrors TensorFlow's PartialTensorShape
// (tensorflow/core/framework/tensor_shape.h), and the Dim functions mirror the
// dimension operations of TensorFlow's InferenceContext
// (tensorflow/core/framework/shape_inference.cc).

#ifndef XLA_SERVING_DATA_PROCESSOR_CC_API_SHAPE_H_
#define XLA_SERVING_DATA_PROCESSOR_CC_API_SHAPE_H_

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <ostream>
#include <string>
#include <utility>

#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "xla/python/ifrt/shape.h"  // from @xla

namespace xla_serving::data {
namespace composite {
namespace api {

// One dimension of a PartialShape: a static size, or unknown.
//
// Unlike in TensorFlow, where PartialTensorShape stores int64 with -1 for
// unknown, here we use a value type Dim to represent a known or unknown value
// for each dimension.
//
// The representation is private on purpose. An abstract API allows extension
// later (like adding a symbolic dimension) without changing its users.
class Dim {
 public:
  // A static size known at compile time, as TF's MakeDim: `Dim::Static(1)`.
  // `size` must be a constant expression, and a negative one doesn't compile.
  // Use FromSize for a size computed at run time.
  static consteval Dim Static(int64_t size) {
    // static_assert can't read a function parameter, even in a consteval
    // function, so a negative size fails on this non-constexpr call instead.
    if (size < 0) NegativeSizeDisallowedInDimStatic();
    return Dim(size);
  }

  // A static size, or InvalidArgument if `size < 0`. A negative size is never
  // read as unknown.
  static absl::StatusOr<Dim> FromSize(int64_t size);

  // An unknown size, as TF's UnknownDim.
  static Dim Unknown() { return Dim(kUnknownRep); }

  // Returns true iff the size is known, as TF's ValueKnown.
  bool IsKnown() const { return rep_ != kUnknownRep; }

  // Returns the size, or nullopt if it is unknown, where TF returns -1.
  // `d.Value() == 1` asks whether `d` is provably 1.
  std::optional<int64_t> Value() const {
    if (!IsKnown()) return std::nullopt;
    return rep_;
  }

  // For error messages: "4", or "?" if unknown.
  std::string DebugString() const;

  // InferenceContext's dimension operations, with its names and semantics, as
  // static functions, so that a shape rule can also run outside a shape
  // function.

  // Merge keeps what either side knows: `?` and 4 merge to 4, and `?` and `?`
  // merge to `?`.
  // Returns InvalidArgument if both sizes are known to be different.
  static absl::StatusOr<Dim> Merge(const Dim& a, const Dim& b);

  // Relax keeps only what both sides agree on: two equal sizes (e.g. 4 and 4)
  // give that size; anything else (different sizes, or `?` on either side)
  // gives `?`.
  static Dim Relax(const Dim& a, const Dim& b);

  // Arithmetic. An unknown operand gives an unknown result, except for TF's
  // special cases: x + 0 = x, x * 0 = 0, x * 1 = x, x / 1 = x and Min(x, 0) =
  // 0. A result that can't be a size is an InvalidArgument error, never a -1.

  // Returns InvalidArgument if the sum overflows int64_t.
  static absl::StatusOr<Dim> Add(const Dim& a, const Dim& b);

  // Returns InvalidArgument if `b` is provably larger than `a`.
  static absl::StatusOr<Dim> Subtract(const Dim& a, const Dim& b);

  // Returns InvalidArgument if the product overflows int64_t.
  static absl::StatusOr<Dim> Multiply(const Dim& a, const Dim& b);

  // Returns InvalidArgument if `divisor <= 0`, or if `evenly_divisible` is true
  // and `a` is known and not a multiple of `divisor`. Unlike TF, a non-positive
  // divisor is an error even if `a` is unknown.
  static absl::StatusOr<Dim> Divide(const Dim& a, int64_t divisor,
                                    bool evenly_divisible);

  // The smaller size: 0 if either is 0, even if the other is unknown.
  static Dim Min(const Dim& a, const Dim& b);

  // The larger size: unknown if either is unknown.
  static Dim Max(const Dim& a, const Dim& b);

  // Structural: ? == ? is true and ? == 4 is false, as when TF compares the
  // int64 dims of two PartialTensorShapes. Two unknown dims compare equal even
  // if they differ at run time; see PartialShape::IsIdenticalTo.
  friend bool operator==(const Dim& a, const Dim& b) {
    return a.rep_ == b.rep_;
  }

  template <typename H>
  friend H AbslHashValue(H h, const Dim& d) {
    return H::combine(std::move(h), d.rep_);
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Dim& d) {
    sink.Append(d.DebugString());
  }

 private:
  // Not a size: every size is non-negative.
  static constexpr int64_t kUnknownRep = std::numeric_limits<int64_t>::min();

  // PartialShape builds Dims from sizes it has already checked.
  friend class PartialShape;

  constexpr explicit Dim(int64_t rep) : rep_(rep) {}

  // Not constexpr, so that Static fails to compile for a negative size. The
  // compiler's error names this function.
  static void NegativeSizeDisallowedInDimStatic() {}

  int64_t rep_;
};

std::ostream& operator<<(std::ostream& os, const Dim& d);

// The partially known shape of a tensor: a static rank, and a Dim for each
// dimension.
//
// Combination of requirements for tensorflow::PartialTensorShape and
// xla::ifrt::Shape:
//   1. No unknown rank, unlike TF's PartialTensorShape(), and no default
//      constructor, like xla::ifrt::Shape: a scalar is xla::ifrt::Shape({}).
//   2. Dimensions are Dims, not int64, so num_elements() returns a Dim,
//      which can be unknown.
//   3. Immutable, like xla::ifrt::Shape. There are no mutators. Code builds a
//      Dimensions and calls Create, which returns InvalidArgument instead of
//      breaking the invariant.
//   4. The fully known shape is xla::ifrt::Shape.
//   5. No implicit constructor from the proto.
//
// For example, ExpandDims inserts a dimension of size 1:
//
//   PartialShape::Dimensions dims(in.dims().begin(), in.dims().end());
//   dims.insert(dims.begin() + axis, Dim::Static(1));
//   ASSIGN_OR_RETURN(PartialShape out, PartialShape::Create(dims));
//
// Invariant: the product of the known, nonzero sizes fits in int64_t, so no
// product of known sizes overflows, in any order. TF's BuildTensorShapeBase
// checks the same. Create and FromProto return InvalidArgument instead of
// breaking it. The conversion from xla::ifrt::Shape requires it.
//
// `==` and IsIdenticalTo are structural, as in TF: [?] == [?], although the two
// dimensions may differ at run time. They are for tests, containers and caches.
// Shape rules use Dim::Value() to ask "provably k", and MergeWith, RelaxWith,
// BroadcastWith and IsCompatibleWith for everything else.
class PartialShape {
 public:
  // Up to rank 6 without allocating, like xla::ifrt::Shape::Dimensions.
  using Dimensions = absl::InlinedVector<Dim, 6>;

  // Returns the number of elements: 0 if any dimension is 0, else unknown if
  // any dimension is unknown, else the product of the sizes.
  Dim num_elements() const;

  // For error messages: "[?,4]".
  std::string DebugString() const;

  // Converts a static shape, as TF's TensorShape converts to
  // PartialTensorShape. REQUIRES: every size is non-negative, and the
  // invariant above.
  PartialShape(const xla::ifrt::Shape& shape);  // NOLINT

  // Returns the dimensions, as xla::ifrt::Shape::dims() does. TF's dims()
  // returns the rank, which here is dims().size().
  absl::Span<const Dim> dims() const { return dims_; }

  // Returns true iff every dimension is known.
  bool IsFullyKnown() const;

  // Returns `*this` as a proto. An unknown dimension has no size.
  PartialShapeProto ToProto() const;

  // Returns a shape with dimensions `dims`, or InvalidArgument if they break
  // the invariant above. This is the only way to build a shape from Dims,
  // computed or written out: copy or build a Dimensions, edit it with plain
  // vector operations, then call Create.
  static absl::StatusOr<PartialShape> Create(absl::Span<const Dim> dims);

  // Like the above, for a brace list whose elements may be errors, so that
  // sizes computed at run time need no separate checks:
  // `Create({Dim::FromSize(m), Dim::Unknown(), Dim::Static(3)})`. Returns the
  // first error, naming its dimension. Every brace list, including one of
  // plain Dims, calls this overload.
  static absl::StatusOr<PartialShape> Create(
      std::initializer_list<absl::StatusOr<Dim>> dims);

  // Returns the shape that `proto` describes, or InvalidArgument if a size is
  // negative or the dimensions break the invariant above. An unset size is an
  // unknown dimension.
  static absl::StatusOr<PartialShape> FromProto(const PartialShapeProto& proto);

  // Returns the shape that knows every dimension that `*this` or `shape`
  // knows, applying Dim::Merge to each pair of dimensions: a fully known
  // shape matches the result iff it matches both. For example, [?,2] merged
  // with [3,2] gives [3,2]. Returns InvalidArgument if the ranks differ, if a
  // pair of dimensions is known and differs (no shape matches both, e.g. [2]
  // and [3]), or if the merged shape breaks the invariant above (e.g.
  // [2**40,?] and [?,2**40]).
  absl::StatusOr<PartialShape> MergeWith(const PartialShape& shape) const;

  // Returns the shape that knows only the dimensions on which `*this` and
  // `shape` agree, applying Dim::Relax to each pair of dimensions: every fully
  // known shape that matches either one matches the result. For example, [?,2]
  // relaxed with [3,2] gives [?,2], and [2] with [3] gives [?]. Control flow
  // uses it to combine the shapes of two branches. Returns InvalidArgument only
  // if the ranks differ.
  absl::StatusOr<PartialShape> RelaxWith(const PartialShape& shape) const;

  // Returns the broadcast of `*this` and `shape`, as NumPy and TF's
  // BroadcastBinaryOpOutputShapeFn compute it. Dimensions are aligned from the
  // right, and a missing dimension counts as 1. For each pair: a 1 gives the
  // other dimension; two known sizes must be equal; a known size wins over an
  // unknown one, which must be 1 or the same size at run time; two unknown
  // dimensions give an unknown one. For example, [?,1,3] broadcast with [4,?]
  // gives [?,4,3]. Returns InvalidArgument if two known sizes differ and
  // neither is 1, or if the result breaks the invariant above.
  absl::StatusOr<PartialShape> BroadcastWith(const PartialShape& shape) const;

  // Exact equality test. Returns true iff the ranks match and every pair of
  // dimensions is equal: both unknown, or both known and equal. This is a
  // stronger condition than IsCompatibleWith.
  bool IsIdenticalTo(const PartialShape& shape) const;

  // Returns true iff the ranks match, and the dimensions all either match or
  // one is unknown: whether the two could be the same shape at run time. Also
  // checks a buffer's xla::ifrt::Shape, through the implicit constructor, as
  // TF's EnsureShape kernel does.
  bool IsCompatibleWith(const PartialShape& shape) const;

  // Returns the static shape, or InvalidArgument naming the first unknown
  // dimension.
  absl::StatusOr<xla::ifrt::Shape> GetStatic() const;

  template <typename H>
  friend H AbslHashValue(H h, const PartialShape& s) {
    return H::combine(std::move(h), s.dims_);
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const PartialShape& s) {
    sink.Append(s.DebugString());
  }

 private:
  // Create checks the invariant.
  explicit PartialShape(Dimensions dims) : dims_(std::move(dims)) {}

  Dimensions dims_;
};

inline bool operator==(const PartialShape& a, const PartialShape& b) {
  return a.IsIdenticalTo(b);
}

std::ostream& operator<<(std::ostream& os, const PartialShape& shape);

}  // namespace api
}  // namespace composite
}  // namespace xla_serving::data

#endif  // XLA_SERVING_DATA_PROCESSOR_CC_API_SHAPE_H_
