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

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <sstream>
#include <type_traits>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_set.h"  // from @abseil-cpp
#include "absl/hash/hash_testing.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "data/test_utils/shape.h"  // from @xla_serving
#include "data/test_utils/status_macros.h"  // from @xla_serving
#include "xla/python/ifrt/shape.h"  // from @xla

namespace xla_serving::data {
namespace composite {
namespace api {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::xla_serving::data::test_utils::PartialShapeOrDie;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Optional;

constexpr int64_t kInt64Max = std::numeric_limits<int64_t>::max();
// Fits in int64_t, but its square does not.
constexpr int64_t kBig = int64_t{1} << 40;

constexpr char kTooLarge[] = "is too large (more than 2**63 - 1 entries)";

consteval Dim S(int64_t v) { return Dim::Static(v); }
Dim U() { return Dim::Unknown(); }
PartialShape Scalar() { return xla::ifrt::Shape({}); }

// The sample set for the lattice laws.
std::vector<Dim> Samples() { return {U(), S(0), S(1), S(4), S(5)}; }

// Ports of tensorflow/core/framework/partial_tensor_shape_test.cc. The unknown
// rank cases, Concatenate, IsValid and MakePartialShape are not carried over.

// Like xla::ifrt::Shape, there is no default constructor. TF's
// PartialTensorShape() has unknown rank.
static_assert(!std::is_default_constructible_v<PartialShape>);

TEST(PartialShapeTest, Scalar) {
  const PartialShape s = xla::ifrt::Shape({});
  EXPECT_EQ(s.dims().size(), 0);
  EXPECT_TRUE(s.IsFullyKnown());
  EXPECT_EQ(s.num_elements(), S(1));
}

TEST(PartialShapeTest, FromProtoAcceptsValidShapes) {
  PartialShapeProto proto;
  EXPECT_OK(PartialShape::FromProto(proto));

  proto.add_dims()->set_size(357);
  proto.add_dims()->set_size(982);
  EXPECT_OK(PartialShape::FromProto(proto));

  // An unknown dimension leaves `size` unset, where TF sets it to -1.
  proto.Clear();
  proto.add_dims()->set_size(0);
  proto.add_dims();
  EXPECT_OK(PartialShape::FromProto(proto));

  // So -1 is as invalid as -2.
  proto.Clear();
  proto.add_dims()->set_size(-1);
  EXPECT_THAT(
      PartialShape::FromProto(proto),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("negative size")));

  proto.Clear();
  proto.add_dims()->set_size(-2);
  EXPECT_THAT(
      PartialShape::FromProto(proto),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("negative size")));
}

TEST(PartialShapeTest, FromProtoRejectsInvalidShapes) {
  PartialShapeProto proto = Scalar().ToProto();
  EXPECT_OK(PartialShape::FromProto(proto));

  proto.add_dims();
  proto.add_dims()->set_size(-2);
  EXPECT_THAT(
      PartialShape::FromProto(proto),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Dimension 1 has a negative size -2; leave `size` "
                         "unset for an unknown dimension")));

  // Unknown dimensions don't count toward the number of elements.
  proto.Clear();
  proto.add_dims()->set_size(kInt64Max);
  proto.add_dims();
  EXPECT_OK(PartialShape::FromProto(proto));

  // Nor do zeros: the product of the nonzero sizes must fit.
  proto.add_dims()->set_size(0);
  proto.add_dims()->set_size(2);
  EXPECT_THAT(PartialShape::FromProto(proto),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr(absl::StrCat(
                           "Shape [9223372036854775807,?,0,2] ", kTooLarge))));

  // A negative size is reported before an overflow.
  proto.add_dims()->set_size(-1);
  EXPECT_THAT(
      PartialShape::FromProto(proto),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("negative size")));
}

TEST(PartialShapeTest, FromProto) {
  const PartialShape s = Scalar();
  const PartialShapeProto sp = s.ToProto();
  ASSERT_OK_AND_ASSIGN(const PartialShape s2, PartialShape::FromProto(sp));
  EXPECT_EQ(s2, s);
  EXPECT_EQ(s2.ToProto().SerializeAsString(), sp.SerializeAsString());

  const PartialShape s3 = PartialShapeOrDie({U(), S(5), S(10)});
  PartialShapeProto sp3 = s3.ToProto();
  ASSERT_OK_AND_ASSIGN(const PartialShape s4, PartialShape::FromProto(sp3));
  EXPECT_EQ(s4, s3);
  EXPECT_EQ(s4.ToProto().SerializeAsString(), sp3.SerializeAsString());

  sp3.add_dims()->set_size(kInt64Max);
  EXPECT_THAT(
      PartialShape::FromProto(sp3),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr(kTooLarge)));

  PartialShapeProto negative;
  negative.add_dims()->set_size(-1);
  EXPECT_THAT(
      PartialShape::FromProto(negative),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("negative size")));

  // An unset size is an unknown dimension.
  PartialShapeProto unset;
  unset.add_dims();
  unset.add_dims()->set_size(3);
  EXPECT_THAT(PartialShape::FromProto(unset),
              IsOkAndHolds(PartialShapeOrDie({U(), S(3)})));
}

TEST(PartialShapeTest, StaticZeroRoundTripsAsKnown) {
  // Presence is explicit, so a static 0 doesn't read back as unknown.
  const PartialShape s = PartialShapeOrDie({S(0), U()});
  const PartialShapeProto proto = s.ToProto();
  ASSERT_EQ(proto.dims_size(), 2);
  EXPECT_TRUE(proto.dims(0).has_size());
  EXPECT_EQ(proto.dims(0).size(), 0);
  EXPECT_FALSE(proto.dims(1).has_size());
  EXPECT_THAT(PartialShape::FromProto(proto), IsOkAndHolds(s));

  PartialShapeProto parsed;
  ASSERT_TRUE(parsed.ParseFromString(proto.SerializeAsString()));
  EXPECT_THAT(PartialShape::FromProto(parsed), IsOkAndHolds(s));
}

TEST(PartialShapeTest, PartialShapeFullyKnown) {
  const PartialShape a = PartialShapeOrDie({U(), S(0), S(1)});
  const PartialShape b = PartialShapeOrDie({S(1), S(0), S(1)});
  const PartialShape c = PartialShapeOrDie({U(), U(), S(1)});
  const PartialShape d = PartialShapeOrDie({S(1), S(0)});
  const PartialShape e = Scalar();
  EXPECT_FALSE(a.IsFullyKnown());
  EXPECT_FALSE(c.IsFullyKnown());
  EXPECT_TRUE(b.IsFullyKnown());
  EXPECT_TRUE(d.IsFullyKnown());
  EXPECT_TRUE(e.IsFullyKnown());
}

TEST(PartialShapeTest, PartialShapeIdenticalTo) {
  const PartialShape a = PartialShapeOrDie({U(), S(0), S(1)});
  const PartialShape b = PartialShapeOrDie({S(1), S(0), S(1)});
  const PartialShape c = PartialShapeOrDie({U(), U(), S(1)});
  const PartialShape d = PartialShapeOrDie({S(1), S(0)});
  const PartialShape e = PartialShapeOrDie({U(), S(0), S(2)});
  const PartialShape f = Scalar();
  const std::vector<PartialShape> shapes = {a, b, c, d, e, f};
  for (size_t i = 0; i < shapes.size(); ++i) {
    for (size_t j = 0; j <= i; ++j) {
      if (i == j) {
        EXPECT_TRUE(shapes[i].IsIdenticalTo(shapes[j]));
      } else {
        EXPECT_FALSE(shapes[i].IsIdenticalTo(shapes[j]));
      }
    }
  }
}

TEST(PartialShapeTest, PartialShapeCompatibleWith) {
  const PartialShape a = PartialShapeOrDie({U(), S(0), S(1)});
  const PartialShape b = PartialShapeOrDie({S(1), S(0), S(1)});
  const PartialShape c = PartialShapeOrDie({U(), U(), S(1)});
  const PartialShape d = PartialShapeOrDie({S(1), S(0)});
  const PartialShape e = PartialShapeOrDie({U(), S(0), S(2)});
  const PartialShape f = Scalar();

  EXPECT_TRUE(f.IsCompatibleWith(f));
  EXPECT_TRUE(a.IsCompatibleWith(b));
  EXPECT_TRUE(a.IsCompatibleWith(a));
  EXPECT_TRUE(b.IsCompatibleWith(b));
  EXPECT_TRUE(a.IsCompatibleWith(c));
  EXPECT_TRUE(b.IsCompatibleWith(c));
  EXPECT_FALSE(a.IsCompatibleWith(d));
  EXPECT_FALSE(b.IsCompatibleWith(d));
  EXPECT_FALSE(c.IsCompatibleWith(d));
  EXPECT_FALSE(a.IsCompatibleWith(e));
  EXPECT_FALSE(b.IsCompatibleWith(e));
  EXPECT_FALSE(c.IsCompatibleWith(e));
  EXPECT_FALSE(a.IsCompatibleWith(f));
  EXPECT_FALSE(b.IsCompatibleWith(f));
  EXPECT_FALSE(c.IsCompatibleWith(f));
}

TEST(PartialShapeTest, ShapeCompatibleWith) {
  const PartialShape a = PartialShapeOrDie({U(), S(0), S(1)});
  const xla::ifrt::Shape b({0, 1});
  const xla::ifrt::Shape c({0, 0, 1});
  const xla::ifrt::Shape d({1, 0, 1});
  const xla::ifrt::Shape e({1, 1, 1});

  EXPECT_FALSE(a.IsCompatibleWith(b));
  EXPECT_TRUE(a.IsCompatibleWith(c));
  EXPECT_TRUE(a.IsCompatibleWith(d));
  EXPECT_FALSE(a.IsCompatibleWith(e));
}

TEST(PartialShapeTest, PartialShapeMergeWith) {
  const PartialShape a = PartialShapeOrDie({U(), S(0), S(1)});
  const PartialShape b = PartialShapeOrDie({S(1), S(0), S(1)});
  const PartialShape c = PartialShapeOrDie({U(), U(), S(1)});
  const PartialShape d = PartialShapeOrDie({S(1), S(0)});

  ASSERT_OK_AND_ASSIGN(PartialShape test, a.MergeWith(a));
  EXPECT_EQ(test.dims().size(), 3);
  EXPECT_EQ(test.dims()[0], U());
  EXPECT_EQ(test.dims()[1], S(0));
  EXPECT_EQ(test.dims()[2], S(1));

  ASSERT_OK_AND_ASSIGN(test, a.MergeWith(b));
  EXPECT_EQ(test.dims().size(), 3);
  EXPECT_EQ(test.dims()[0], S(1));
  EXPECT_EQ(test.dims()[1], S(0));
  EXPECT_EQ(test.dims()[2], S(1));

  EXPECT_THAT(a.MergeWith(d),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("PartialShape: Incompatible ranks during "
                                 "merge: 3 vs. 2")));

  ASSERT_OK_AND_ASSIGN(test, a.MergeWith(c));
  EXPECT_EQ(test.dims().size(), 3);
  EXPECT_EQ(test.dims()[0], U());
  EXPECT_EQ(test.dims()[1], S(0));
  EXPECT_EQ(test.dims()[2], S(1));

  ASSERT_OK_AND_ASSIGN(test, c.MergeWith(a));
  EXPECT_EQ(test.dims().size(), 3);
  EXPECT_EQ(test.dims()[0], U());
  EXPECT_EQ(test.dims()[1], S(0));
  EXPECT_EQ(test.dims()[2], S(1));
}

TEST(PartialShapeTest, PartialShapeMergeWithInvalidData) {
  // TF's "Cannot output result to itself" case doesn't apply: MergeWith
  // returns its result.
  const PartialShape b = PartialShapeOrDie({U(), S(0), S(2)});
  const PartialShape c = PartialShapeOrDie({S(1), U(), S(3)});
  const PartialShape d = PartialShapeOrDie({U(), S(kInt64Max), U()});

  EXPECT_THAT(b.MergeWith(c),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("PartialShape: Incompatible shapes during "
                                 "merge: [?,0,2] vs. [1,?,3]")));
  EXPECT_THAT(c.MergeWith(d),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr(absl::StrCat(
                           "Shape [1,9223372036854775807,3] ", kTooLarge))));

  // Each side satisfies the invariant, but the merged shape would not.
  const PartialShape e = PartialShapeOrDie({S(kBig), U()});
  const PartialShape f = PartialShapeOrDie({U(), S(kBig)});
  EXPECT_THAT(e.MergeWith(f), StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr(kTooLarge)));
}

TEST(PartialShapeTest, ZeroAndDynamicDims) {
  // TF says -1 elements for both. As in InferenceContext, both have 0.
  const PartialShape s1 = PartialShapeOrDie({S(0), U()});
  const PartialShape s2 = PartialShapeOrDie({U(), S(0)});
  EXPECT_EQ(s1.num_elements(), S(0));
  EXPECT_EQ(s2.num_elements(), S(0));

  // A shape with a zero and a large size has 0 elements: no overflow.
  const PartialShape s3 = PartialShapeOrDie({S(0), S(kInt64Max)});
  const PartialShape s4 = PartialShapeOrDie({S(kInt64Max), S(0)});
  EXPECT_EQ(s3.num_elements(), S(0));
  EXPECT_EQ(s4.num_elements(), S(0));
}

// Dim.

TEST(DimTest, Static) {
  EXPECT_TRUE(S(0).IsKnown());
  EXPECT_THAT(S(0).Value(), Optional(0));
  EXPECT_TRUE(S(4).IsKnown());
  EXPECT_THAT(S(4).Value(), Optional(4));
  EXPECT_THAT(S(kInt64Max).Value(), Optional(kInt64Max));
}

TEST(DimTest, Unknown) {
  EXPECT_FALSE(U().IsKnown());
  EXPECT_EQ(U().Value(), std::nullopt);
  // `d.Value() == k` asks whether `d` is provably k.
  EXPECT_NE(U().Value(), 1);
  EXPECT_EQ(S(1).Value(), 1);
}

TEST(DimTest, FromSize) {
  EXPECT_THAT(Dim::FromSize(0), IsOkAndHolds(S(0)));
  EXPECT_THAT(Dim::FromSize(4), IsOkAndHolds(S(4)));
  EXPECT_THAT(Dim::FromSize(kInt64Max), IsOkAndHolds(S(kInt64Max)));
  // A negative size is an error, never an unknown dimension.
  EXPECT_THAT(Dim::FromSize(-1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected a non-negative size, got -1")));
}

TEST(DimTest, DebugString) {
  EXPECT_EQ(S(0).DebugString(), "0");
  EXPECT_EQ(S(4).DebugString(), "4");
  EXPECT_EQ(U().DebugString(), "?");
  EXPECT_EQ(absl::StrCat(S(4), ",", U()), "4,?");
  std::ostringstream os;
  os << S(4) << "," << U();
  EXPECT_EQ(os.str(), "4,?");
}

TEST(DimTest, EqualityIsStructural) {
  EXPECT_EQ(S(4), S(4));
  EXPECT_EQ(U(), U());
  EXPECT_NE(S(4), S(5));
  EXPECT_NE(S(4), U());
  EXPECT_NE(S(0), U());
}

TEST(DimTest, Hash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly(
      {U(), S(0), S(1), S(4), S(kInt64Max)}));
  const absl::flat_hash_set<Dim> dims = {S(4), U(), S(4), U(), S(0)};
  EXPECT_EQ(dims.size(), 3);
  EXPECT_TRUE(dims.contains(U()));
  EXPECT_TRUE(dims.contains(S(0)));
  EXPECT_FALSE(dims.contains(S(1)));
}

// The lattice laws, over Samples().

TEST(DimLatticeTest, UnknownIsMergeIdentity) {
  for (const Dim& a : Samples()) {
    EXPECT_THAT(Dim::Merge(a, U()), IsOkAndHolds(a)) << a;
    EXPECT_THAT(Dim::Merge(U(), a), IsOkAndHolds(a)) << a;
  }
}

TEST(DimLatticeTest, MergeIsIdempotent) {
  for (const Dim& a : Samples()) {
    EXPECT_THAT(Dim::Merge(a, a), IsOkAndHolds(a)) << a;
  }
}

TEST(DimLatticeTest, MergeIsCommutative) {
  for (const Dim& a : Samples()) {
    for (const Dim& b : Samples()) {
      const absl::StatusOr<Dim> ab = Dim::Merge(a, b);
      const absl::StatusOr<Dim> ba = Dim::Merge(b, a);
      ASSERT_EQ(ab.ok(), ba.ok()) << a << ", " << b;
      if (ab.ok()) EXPECT_EQ(*ab, *ba) << a << ", " << b;
    }
  }
}

TEST(DimLatticeTest, MergeIsAssociative) {
  for (const Dim& a : Samples()) {
    for (const Dim& b : Samples()) {
      for (const Dim& c : Samples()) {
        const absl::StatusOr<Dim> ab = Dim::Merge(a, b);
        const absl::StatusOr<Dim> bc = Dim::Merge(b, c);
        const absl::StatusOr<Dim> ab_c = ab.ok() ? Dim::Merge(*ab, c) : ab;
        const absl::StatusOr<Dim> a_bc = bc.ok() ? Dim::Merge(a, *bc) : bc;
        ASSERT_EQ(ab_c.ok(), a_bc.ok()) << a << ", " << b << ", " << c;
        if (ab_c.ok()) EXPECT_EQ(*ab_c, *a_bc) << a << ", " << b << ", " << c;
      }
    }
  }
}

TEST(DimLatticeTest, MergeOfDifferentSizesFails) {
  EXPECT_THAT(Dim::Merge(S(4), S(5)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Dimensions must be equal, but are 4 and 5")));
  EXPECT_THAT(Dim::Merge(S(0), S(1)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Dimensions must be equal, but are 0 and 1")));
}

TEST(DimLatticeTest, UnknownAbsorbsRelax) {
  for (const Dim& a : Samples()) {
    EXPECT_EQ(Dim::Relax(a, U()), U()) << a;
    EXPECT_EQ(Dim::Relax(U(), a), U()) << a;
  }
}

TEST(DimLatticeTest, RelaxIsIdempotent) {
  for (const Dim& a : Samples()) {
    EXPECT_EQ(Dim::Relax(a, a), a) << a;
  }
}

TEST(DimLatticeTest, RelaxIsCommutative) {
  for (const Dim& a : Samples()) {
    for (const Dim& b : Samples()) {
      EXPECT_EQ(Dim::Relax(a, b), Dim::Relax(b, a)) << a << ", " << b;
    }
  }
}

TEST(DimLatticeTest, RelaxIsAssociative) {
  for (const Dim& a : Samples()) {
    for (const Dim& b : Samples()) {
      for (const Dim& c : Samples()) {
        EXPECT_EQ(Dim::Relax(Dim::Relax(a, b), c),
                  Dim::Relax(a, Dim::Relax(b, c)))
            << a << ", " << b << ", " << c;
      }
    }
  }
}

TEST(DimLatticeTest, RelaxOfDifferentSizesIsUnknown) {
  EXPECT_EQ(Dim::Relax(S(4), S(5)), U());
  EXPECT_EQ(Dim::Relax(S(0), S(1)), U());
  EXPECT_EQ(Dim::Relax(S(4), S(4)), S(4));
}

TEST(DimLatticeTest, Absorption) {
  for (const Dim& a : Samples()) {
    for (const Dim& b : Samples()) {
      if (const absl::StatusOr<Dim> merged = Dim::Merge(a, b); merged.ok()) {
        EXPECT_EQ(Dim::Relax(a, *merged), a) << a << ", " << b;
      }
      EXPECT_THAT(Dim::Merge(a, Dim::Relax(a, b)), IsOkAndHolds(a))
          << a << ", " << b;
    }
  }
}

// Arithmetic, including TF's special cases.

TEST(DimArithmeticTest, Add) {
  EXPECT_THAT(Dim::Add(S(3), S(4)), IsOkAndHolds(S(7)));
  EXPECT_THAT(Dim::Add(S(4), S(0)), IsOkAndHolds(S(4)));
  EXPECT_THAT(Dim::Add(U(), S(0)), IsOkAndHolds(U()));
  EXPECT_THAT(Dim::Add(S(0), S(4)), IsOkAndHolds(S(4)));
  EXPECT_THAT(Dim::Add(S(0), U()), IsOkAndHolds(U()));
  EXPECT_THAT(Dim::Add(U(), S(4)), IsOkAndHolds(U()));
  EXPECT_THAT(Dim::Add(S(4), U()), IsOkAndHolds(U()));
  EXPECT_THAT(Dim::Add(S(kInt64Max), S(1)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Dimension size overflow from adding "
                                 "9223372036854775807 and 1")));
}

TEST(DimArithmeticTest, Subtract) {
  EXPECT_THAT(Dim::Subtract(S(4), S(3)), IsOkAndHolds(S(1)));
  EXPECT_THAT(Dim::Subtract(S(4), S(4)), IsOkAndHolds(S(0)));
  EXPECT_THAT(Dim::Subtract(S(4), S(0)), IsOkAndHolds(S(4)));
  EXPECT_THAT(Dim::Subtract(U(), S(0)), IsOkAndHolds(U()));
  EXPECT_THAT(Dim::Subtract(U(), S(4)), IsOkAndHolds(U()));
  EXPECT_THAT(Dim::Subtract(S(4), U()), IsOkAndHolds(U()));
  EXPECT_THAT(Dim::Subtract(S(3), S(4)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Negative dimension size caused by "
                                 "subtracting 4 from 3")));
}

TEST(DimArithmeticTest, Multiply) {
  EXPECT_THAT(Dim::Multiply(S(3), S(4)), IsOkAndHolds(S(12)));
  EXPECT_THAT(Dim::Multiply(S(4), S(0)), IsOkAndHolds(S(0)));
  EXPECT_THAT(Dim::Multiply(U(), S(0)), IsOkAndHolds(S(0)));
  EXPECT_THAT(Dim::Multiply(S(0), U()), IsOkAndHolds(S(0)));
  EXPECT_THAT(Dim::Multiply(S(4), S(1)), IsOkAndHolds(S(4)));
  EXPECT_THAT(Dim::Multiply(U(), S(1)), IsOkAndHolds(U()));
  EXPECT_THAT(Dim::Multiply(S(1), U()), IsOkAndHolds(U()));
  EXPECT_THAT(Dim::Multiply(U(), S(4)), IsOkAndHolds(U()));
  EXPECT_THAT(Dim::Multiply(S(4), U()), IsOkAndHolds(U()));
  EXPECT_THAT(Dim::Multiply(S(int64_t{1} << 32), S(int64_t{1} << 31)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Dimension size overflow from multiplying "
                                 "4294967296 and 2147483648")));
}

TEST(DimArithmeticTest, Divide) {
  // x / 1 = x, known or not.
  EXPECT_THAT(Dim::Divide(U(), 1, /*evenly_divisible=*/true),
              IsOkAndHolds(U()));
  EXPECT_THAT(Dim::Divide(S(7), 1, /*evenly_divisible=*/true),
              IsOkAndHolds(S(7)));
  EXPECT_THAT(Dim::Divide(S(7), 1, /*evenly_divisible=*/false),
              IsOkAndHolds(S(7)));
  EXPECT_THAT(Dim::Divide(S(8), 2, /*evenly_divisible=*/true),
              IsOkAndHolds(S(4)));
  EXPECT_THAT(Dim::Divide(S(9), 3, /*evenly_divisible=*/false),
              IsOkAndHolds(S(3)));
  // 0 is evenly divisible by anything.
  EXPECT_THAT(Dim::Divide(S(0), 3, /*evenly_divisible=*/true),
              IsOkAndHolds(S(0)));
  EXPECT_THAT(Dim::Divide(S(7), 2, /*evenly_divisible=*/true),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Dimension size must be evenly divisible by "
                                 "2 but is 7")));
  EXPECT_THAT(Dim::Divide(S(7), 2, /*evenly_divisible=*/false),
              IsOkAndHolds(S(3)));
  EXPECT_THAT(Dim::Divide(U(), 2, /*evenly_divisible=*/true),
              IsOkAndHolds(U()));
  // A non-positive divisor is an error, even if the dividend is unknown.
  EXPECT_THAT(Dim::Divide(S(8), 0, /*evenly_divisible=*/false),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Divisor must be positive but is 0")));
  EXPECT_THAT(Dim::Divide(U(), 0, /*evenly_divisible=*/false),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Divisor must be positive but is 0")));
  EXPECT_THAT(Dim::Divide(U(), -1, /*evenly_divisible=*/false),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Divisor must be positive but is -1")));
}

TEST(DimArithmeticTest, Min) {
  EXPECT_EQ(Dim::Min(U(), S(0)), S(0));
  EXPECT_EQ(Dim::Min(S(0), U()), S(0));
  EXPECT_EQ(Dim::Min(U(), S(4)), U());
  EXPECT_EQ(Dim::Min(S(4), U()), U());
  EXPECT_EQ(Dim::Min(S(3), S(4)), S(3));
  EXPECT_EQ(Dim::Min(S(4), S(3)), S(3));
}

TEST(DimArithmeticTest, Max) {
  EXPECT_EQ(Dim::Max(U(), S(0)), U());
  EXPECT_EQ(Dim::Max(S(0), U()), U());
  EXPECT_EQ(Dim::Max(S(3), S(4)), S(4));
  EXPECT_EQ(Dim::Max(S(4), S(3)), S(4));
}

// PartialShape.

TEST(PartialShapeTest, NumElements) {
  EXPECT_EQ(PartialShapeOrDie({S(0), U()}).num_elements(), S(0));
  EXPECT_EQ(PartialShapeOrDie({U(), S(4)}).num_elements(), U());
  EXPECT_EQ(PartialShapeOrDie({S(2), S(3)}).num_elements(), S(6));
  EXPECT_EQ(Scalar().num_elements(), S(1));
}

TEST(PartialShapeTest, DebugString) {
  EXPECT_EQ(Scalar().DebugString(), "[]");
  EXPECT_EQ(PartialShapeOrDie({U(), S(4)}).DebugString(), "[?,4]");
  EXPECT_EQ(absl::StrCat(PartialShapeOrDie({U(), S(4)})), "[?,4]");
  std::ostringstream os;
  os << PartialShapeOrDie({S(2), U()});
  EXPECT_EQ(os.str(), "[2,?]");
}

TEST(PartialShapeTest, EqualityIsStructural) {
  EXPECT_EQ(PartialShapeOrDie({U()}), PartialShapeOrDie({U()}));
  EXPECT_NE(PartialShapeOrDie({U()}), PartialShapeOrDie({S(4)}));
  EXPECT_NE(PartialShapeOrDie({S(4)}), PartialShapeOrDie({S(4), S(1)}));
  EXPECT_NE(Scalar(), PartialShapeOrDie({S(1)}));
  // Through the implicit constructor from xla::ifrt::Shape.
  EXPECT_EQ(PartialShapeOrDie({S(2), S(3)}), xla::ifrt::Shape({2, 3}));
  EXPECT_EQ(Scalar(), PartialShapeOrDie({}));
}

TEST(PartialShapeTest, Hash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly(
      {Scalar(), PartialShapeOrDie({U()}), PartialShapeOrDie({S(4)}),
       PartialShapeOrDie({U(), U()}), PartialShapeOrDie({U(), S(4)}),
       PartialShapeOrDie({S(4), U()})}));
  const absl::flat_hash_set<PartialShape> shapes = {
      PartialShapeOrDie({U(), S(4)}), PartialShapeOrDie({U(), S(4)}),
      PartialShapeOrDie({S(4)}), Scalar(), Scalar()};
  EXPECT_EQ(shapes.size(), 3);
  EXPECT_TRUE(shapes.contains(PartialShapeOrDie({U(), S(4)})));
  EXPECT_FALSE(shapes.contains(PartialShapeOrDie({S(4), U()})));
}

// Create is the only way to build a shape from Dims.
static_assert(
    !std::is_constructible_v<PartialShape, std::initializer_list<Dim>>);
static_assert(!std::is_constructible_v<PartialShape, absl::Span<const Dim>>);
static_assert(!std::is_constructible_v<PartialShape, PartialShape::Dimensions>);

TEST(PartialShapeTest, Create) {
  // From a brace list, a Dimensions, or a sub-span of another shape's dims().
  EXPECT_THAT(PartialShape::Create({S(2), U()}),
              IsOkAndHolds(PartialShapeOrDie({S(2), U()})));
  PartialShape::Dimensions dims = {U(), S(0)};
  dims.push_back(S(3));
  EXPECT_THAT(PartialShape::Create(dims),
              IsOkAndHolds(PartialShapeOrDie({U(), S(0), S(3)})));
  const PartialShape s = PartialShapeOrDie({S(2), U(), S(3)});
  EXPECT_THAT(PartialShape::Create(s.dims().subspan(1)),
              IsOkAndHolds(PartialShapeOrDie({U(), S(3)})));
  EXPECT_THAT(PartialShape::Create({}), IsOkAndHolds(Scalar()));

  // Returns InvalidArgument instead of breaking the invariant.
  EXPECT_THAT(
      PartialShape::Create({S(kBig), S(kBig)}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("too large")));
  EXPECT_THAT(
      PartialShape::Create({S(kBig), U(), S(kBig)}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr(absl::StrCat("Shape [1099511627776,?,1099511627776] ",
                                      kTooLarge))));
  // Unknown dimensions and zeros don't count, but a zero doesn't hide an
  // overflow of the other sizes.
  EXPECT_THAT(PartialShape::Create({S(kBig), U(), S(0)}),
              IsOkAndHolds(PartialShapeOrDie({S(kBig), U(), S(0)})));
  EXPECT_THAT(
      PartialShape::Create({S(kBig), S(0), S(kBig)}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr(kTooLarge)));
}

TEST(PartialShapeTest, CreateFromStatusOrDims) {
  // A brace list may mix Dims and the results of Dim::FromSize.
  EXPECT_THAT(PartialShape::Create({Dim::FromSize(2), U(), S(3)}),
              IsOkAndHolds(PartialShapeOrDie({S(2), U(), S(3)})));

  // The first error, naming its dimension.
  EXPECT_THAT(
      PartialShape::Create({S(2), Dim::FromSize(-3), Dim::FromSize(-4)}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Dimension 1: Expected a non-negative size, got -3")));

  // The invariant is still checked.
  EXPECT_THAT(
      PartialShape::Create({Dim::FromSize(kBig), Dim::FromSize(kBig)}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr(kTooLarge)));
}

TEST(PartialShapeTest, Dims) {
  // The dimensions, as in xla::ifrt::Shape. The rank is dims().size().
  const PartialShape s = PartialShapeOrDie({S(2), U()});
  EXPECT_THAT(s.dims(), ElementsAre(S(2), U()));
  EXPECT_EQ(s.dims().size(), 2);
  const PartialShape scalar = Scalar();
  EXPECT_THAT(scalar.dims(), IsEmpty());
}

TEST(PartialShapeTest, ExpandDims) {
  // The idiom from the class comment, with ASSERT_OK_AND_ASSIGN for
  // ASSIGN_OR_RETURN.
  const PartialShape in = PartialShapeOrDie({S(2), U()});
  const std::vector<PartialShape> expected = {
      PartialShapeOrDie({S(1), S(2), U()}),
      PartialShapeOrDie({S(2), S(1), U()}),
      PartialShapeOrDie({S(2), U(), S(1)})};
  for (int axis = 0; axis <= 2; ++axis) {
    PartialShape::Dimensions dims(in.dims().begin(), in.dims().end());
    dims.insert(dims.begin() + axis, Dim::Static(1));
    ASSERT_OK_AND_ASSIGN(PartialShape out, PartialShape::Create(dims));
    EXPECT_EQ(out, expected[axis]) << axis;
  }
}

TEST(PartialShapeTest, ToProto) {
  const PartialShape s = PartialShapeOrDie({S(2), U(), S(0)});
  const PartialShapeProto proto = s.ToProto();
  ASSERT_EQ(proto.dims_size(), 3);
  EXPECT_TRUE(proto.dims(0).has_size());
  EXPECT_EQ(proto.dims(0).size(), 2);
  EXPECT_FALSE(proto.dims(1).has_size());
  EXPECT_TRUE(proto.dims(2).has_size());
  EXPECT_EQ(proto.dims(2).size(), 0);
  EXPECT_THAT(PartialShape::FromProto(proto), IsOkAndHolds(s));
}

TEST(PartialShapeTest, ProtoRoundTrip) {
  // Static zeros stay known, and unknown dimensions stay unknown.
  const std::vector<PartialShape> shapes = {
      Scalar(), PartialShapeOrDie({S(0)}), PartialShapeOrDie({U()}),
      PartialShapeOrDie({S(2), U(), S(0)}),
      PartialShapeOrDie({S(kInt64Max), U(), S(0)})};
  for (const PartialShape& s : shapes) {
    EXPECT_THAT(PartialShape::FromProto(s.ToProto()), IsOkAndHolds(s)) << s;
  }
}

TEST(PartialShapeTest, GetStatic) {
  ASSERT_OK_AND_ASSIGN(const xla::ifrt::Shape shape,
                       PartialShapeOrDie({S(2), S(3)}).GetStatic());
  EXPECT_THAT(shape.dims(), ElementsAre(2, 3));
  ASSERT_OK_AND_ASSIGN(const xla::ifrt::Shape scalar, Scalar().GetStatic());
  EXPECT_THAT(scalar.dims(), IsEmpty());
  ASSERT_OK_AND_ASSIGN(const xla::ifrt::Shape empty,
                       PartialShapeOrDie({S(1), S(0)}).GetStatic());
  EXPECT_THAT(empty.dims(), ElementsAre(1, 0));

  EXPECT_THAT(PartialShapeOrDie({S(2), U()}).GetStatic(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Shape [2,?] is not fully known: "
                                 "dimension 1 is unknown")));
}

TEST(PartialShapeTest, RelaxWith) {
  EXPECT_THAT(PartialShapeOrDie({S(2), S(3)})
                  .RelaxWith(PartialShapeOrDie({S(2), S(4)})),
              IsOkAndHolds(PartialShapeOrDie({S(2), U()})));
  EXPECT_THAT(
      PartialShapeOrDie({U(), S(3)}).RelaxWith(PartialShapeOrDie({S(2), S(3)})),
      IsOkAndHolds(PartialShapeOrDie({U(), S(3)})));
  EXPECT_THAT(Scalar().RelaxWith(Scalar()), IsOkAndHolds(Scalar()));
  EXPECT_THAT(
      PartialShapeOrDie({S(2)}).RelaxWith(PartialShapeOrDie({S(2), S(4)})),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Cannot relax shapes of different ranks: "
                         "[2] vs. [2,4]")));
}

TEST(PartialShapeTest, BroadcastWith) {
  // Static shapes broadcast as in NumPy; a missing dimension counts as 1.
  EXPECT_THAT(
      PartialShapeOrDie({S(2), S(1)}).BroadcastWith(PartialShapeOrDie({S(3)})),
      IsOkAndHolds(PartialShapeOrDie({S(2), S(3)})));
  EXPECT_THAT(Scalar().BroadcastWith(PartialShapeOrDie({U(), S(4)})),
              IsOkAndHolds(PartialShapeOrDie({U(), S(4)})));
  // A 1 gives the other dimension, unknown or not.
  EXPECT_THAT(PartialShapeOrDie({S(1)}).BroadcastWith(PartialShapeOrDie({U()})),
              IsOkAndHolds(PartialShapeOrDie({U()})));
  // A known size other than 1 wins over an unknown one.
  EXPECT_THAT(PartialShapeOrDie({U(), S(1), S(3)})
                  .BroadcastWith(PartialShapeOrDie({S(4), U()})),
              IsOkAndHolds(PartialShapeOrDie({U(), S(4), S(3)})));
  EXPECT_THAT(PartialShapeOrDie({S(0)}).BroadcastWith(PartialShapeOrDie({U()})),
              IsOkAndHolds(PartialShapeOrDie({S(0)})));
  EXPECT_THAT(PartialShapeOrDie({U()}).BroadcastWith(PartialShapeOrDie({U()})),
              IsOkAndHolds(PartialShapeOrDie({U()})));
  // Two equal known sizes give that size.
  EXPECT_THAT(PartialShapeOrDie({S(2), U()})
                  .BroadcastWith(PartialShapeOrDie({S(2), S(3)})),
              IsOkAndHolds(PartialShapeOrDie({S(2), S(3)})));
  // Two known sizes must be equal unless one is 1.
  EXPECT_THAT(
      PartialShapeOrDie({S(2)}).BroadcastWith(PartialShapeOrDie({S(3)})),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Incompatible shapes: [2] vs. [3]")));
  // The result must satisfy the invariant.
  EXPECT_THAT(
      PartialShapeOrDie({S(kBig), S(1)})
          .BroadcastWith(PartialShapeOrDie({S(1), S(kBig)})),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr(kTooLarge)));
}

TEST(PartialShapeDeathTest, ConversionFromIfrtShapeRequiresTheInvariant) {
  EXPECT_DEBUG_DEATH(
      { const PartialShape shape(xla::ifrt::Shape({kBig, kBig})); },
      "too large");
}

}  // namespace
}  // namespace api
}  // namespace composite
}  // namespace xla_serving::data
