/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Copyright 2026 Google LLC. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "data/kernels/op_common.h"  // from @xla_serving

#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "data/kernels/bitwise/op_functors.h"  // from @xla_serving
#include "data/kernels/math/op_functors.h"  // from @xla_serving
#include "data/kernels/test_utils.h"  // from @xla_serving
#include "data/processor/cc/api/shape.h"  // from @xla_serving
#include "data/processor/cc/api/transform_context.h"  // from @xla_serving
#include "data/test_utils/shape.h"  // from @xla_serving
#include "data/test_utils/status_macros.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "tsl/platform/tstring.h"  // from @tsl

namespace xla_serving::data::kernel {
namespace {

using ::absl_testing::StatusIs;
using ::xla_serving::data::composite::api::Dim;
using ::xla_serving::data::test_utils::PartialShapeOrDie;
using ::testing::HasSubstr;
using ::xla::ifrt::DType;
using ::xla::ifrt::Shape;

// Test functors for validating generic bases
struct TestNegateFunctor {
  template <typename T>
    requires(!std::is_same_v<T, tsl::tstring>)
  T operator()(T x) const {
    return -x;
  }
};

struct TestAddFunctor {
  template <typename T>
  T operator()(T a, T b) const {
    return a + b;
  }
};

struct TestCheckedDivFunctor {
  template <typename T>
    requires(!std::is_same_v<T, bool> && !std::is_same_v<T, tsl::tstring>)
  absl::StatusOr<T> operator()(T a, T b) const {
    if (b == T(0)) {
      return absl::InvalidArgumentError("Division by zero");
    }
    return a / b;
  }
};

ND_DECLARE_UNARY_OP(TestNegateTransform, TestNegateFunctor);
ND_DECLARE_BINARY_OP(TestAddTransform, TestAddFunctor);
ND_DECLARE_BINARY_OP(TestCheckedDivTransform, TestCheckedDivFunctor);

TEST(OpCommonTest, TestUnaryOpExecution) {
  auto a = CreateBuffer<float>({2, 1}, {4.0f, -16.0f});
  ASSERT_OK_AND_ASSIGN(auto transform, TestNegateTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a)));

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2, 1}));
  EXPECT_EQ(GetData<float>(out), std::vector<float>({-4.0f, 16.0f}));
}

TEST(OpCommonTest, TestNegateTransformStringReturnsUnimplemented) {
  auto str_buf = CreateBuffer<tsl::tstring>({2}, {"a", "b"});
  ASSERT_OK_AND_ASSIGN(auto transform, TestNegateTransform::Create());
  EXPECT_THAT(
      (*transform)(std::move(str_buf)),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("Operator 'TestNegateTransform' does not provide an "
                         "implementation for dtype: STRING")));
}

TEST(OpCommonTest, TestBinaryOpExecution) {
  auto a = CreateBuffer<int32_t>({2}, {1, 2});
  auto b = CreateBuffer<int32_t>({2}, {3, 4});
  ASSERT_OK_AND_ASSIGN(auto transform, TestAddTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.dtype(), DType(DType::kS32));
  EXPECT_EQ(out.shape(), Shape({2}));
  EXPECT_EQ(GetData<int32_t>(out), std::vector<int32_t>({4, 6}));
}

TEST(OpCommonTest, TestStatusOrReturningFunctorError) {
  auto a = CreateBuffer<int32_t>({2}, {5, 2});
  auto b = CreateBuffer<int32_t>({2}, {0, 1});
  ASSERT_OK_AND_ASSIGN(auto transform, TestCheckedDivTransform::Create());
  EXPECT_THAT((*transform)(std::move(a), std::move(b)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Division by zero")));
}

TEST(OpCommonTest, TestShapeMismatchError) {
  auto a = CreateBuffer<int32_t>({2}, {1, 2});
  auto b = CreateBuffer<int32_t>({3}, {1, 2, 3});
  ASSERT_OK_AND_ASSIGN(auto transform, TestAddTransform::Create());
  EXPECT_THAT(
      (*transform)(std::move(a), std::move(b)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("TestAddTransform inputs are not broadcast-compatible."
                         " Got [2] and [3]")));
}

TEST(OpCommonTest, TestDTypeMismatchError) {
  auto a = CreateBuffer<int32_t>({2}, {1, 2});
  auto b = CreateBuffer<float>({2}, {1.0f, 2.0f});
  ASSERT_OK_AND_ASSIGN(auto transform, TestAddTransform::Create());
  EXPECT_THAT((*transform)(std::move(a), std::move(b)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("TestAddTransform inputs must have the same "
                                 "dtype. Got S32 and F32")));
}

struct TestFloatOnlyFunctor {
  template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
  T operator()(T x) const {
    return x * 2.0f;
  }
};
ND_DECLARE_UNARY_OP(TestFloatOnlyTransform, TestFloatOnlyFunctor);

TEST(OpCommonTest, TestUnsupportedDTypeReturnsUnimplemented) {
  ASSERT_OK_AND_ASSIGN(auto transform, TestFloatOnlyTransform::Create());

  auto a = CreateBuffer<int32_t>({2}, {1, 2});
  EXPECT_THAT(
      (*transform)(std::move(a)),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("Operator 'TestFloatOnlyTransform' does not provide "
                         "an implementation for dtype: S32")));

  auto str_buf = CreateBuffer<tsl::tstring>({2}, {"a", "b"});
  EXPECT_THAT(
      (*transform)(std::move(str_buf)),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("Operator 'TestFloatOnlyTransform' does not provide "
                         "an implementation for dtype: STRING")));
}

struct TestFloatOnlyBinaryFunctor {
  template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
  T operator()(T a, T b) const {
    return a * b;
  }
};
ND_DECLARE_BINARY_OP(TestFloatOnlyBinaryTransform, TestFloatOnlyBinaryFunctor);

TEST(OpCommonTest, TestUnsupportedBinaryDTypeReturnsUnimplemented) {
  ASSERT_OK_AND_ASSIGN(auto transform, TestFloatOnlyBinaryTransform::Create());

  auto a = CreateBuffer<int32_t>({2}, {1, 2});
  auto b = CreateBuffer<int32_t>({2}, {3, 4});
  EXPECT_THAT(
      (*transform)(std::move(a), std::move(b)),
      StatusIs(
          absl::StatusCode::kUnimplemented,
          HasSubstr("Operator 'TestFloatOnlyBinaryTransform' does not provide "
                    "an implementation for dtype: S32")));

  auto str1 = CreateBuffer<tsl::tstring>({2}, {"a", "b"});
  auto str2 = CreateBuffer<tsl::tstring>({2}, {"c", "d"});
  EXPECT_THAT(
      (*transform)(std::move(str1), std::move(str2)),
      StatusIs(
          absl::StatusCode::kUnimplemented,
          HasSubstr("Operator 'TestFloatOnlyBinaryTransform' does not provide "
                    "an implementation for dtype: STRING")));
}

TEST(OpCommonTest, TestEmptyTensorExecution) {
  auto a = CreateBuffer<float>({0}, {});
  ASSERT_OK_AND_ASSIGN(auto transform, TestNegateTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a)));

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({0}));
  EXPECT_EQ(out.shape().num_elements(), 0);
}

TEST(OpCommonTest, TestInt64AndFloat64Execution) {
  auto a_i64 = CreateBuffer<int64_t>({2}, {100, 200});
  auto b_i64 = CreateBuffer<int64_t>({2}, {300, 400});
  ASSERT_OK_AND_ASSIGN(auto add_transform, TestAddTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out_i64,
                       (*add_transform)(std::move(a_i64), std::move(b_i64)));

  EXPECT_EQ(out_i64.dtype(), DType(DType::kS64));
  EXPECT_EQ(GetData<int64_t>(out_i64), std::vector<int64_t>({400, 600}));

  auto a_f64 = CreateBuffer<double>({2}, {1.5, -2.5});
  ASSERT_OK_AND_ASSIGN(auto negate_transform, TestNegateTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out_f64, (*negate_transform)(std::move(a_f64)));

  EXPECT_EQ(out_f64.dtype(), DType(DType::kF64));
  EXPECT_EQ(GetData<double>(out_f64), std::vector<double>({-1.5, 2.5}));
}

TEST(OpCommonTest, TestToProtoOpName) {
  ASSERT_OK_AND_ASSIGN(auto unary_transform, TestNegateTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto unary_config, unary_transform->ToProto());
  EXPECT_EQ(unary_config.op_name(), "TestNegateTransform");

  ASSERT_OK_AND_ASSIGN(auto binary_transform, TestAddTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto binary_config, binary_transform->ToProto());
  EXPECT_EQ(binary_config.op_name(), "TestAddTransform");
}

TEST(OpCommonTest, TestPropagateSignatureSuccess) {
  std::vector<xla_serving::data::composite::api::ShapeInferenceValue> inputs;
  inputs.push_back(xla_serving::data::composite::api::HostBufferSpec{
      .dtype = DType(DType::kS32),
      .shape = Shape({2, 3}),
  });
  inputs.push_back(xla_serving::data::composite::api::HostBufferSpec{
      .dtype = DType(DType::kS32),
      .shape = Shape({2, 3}),
  });

  xla_serving::data::composite::api::ShapeInferenceContext context;
  context.inputs = inputs;

  EXPECT_OK(
      BinaryOpShared::PropagateBinarySignature(context, "TestAddTransform"));
  ASSERT_EQ(context.outputs.size(), 1);
  EXPECT_TRUE(context.outputs[0].IsSpec());
  EXPECT_EQ(context.outputs[0].spec().shape, Shape({2, 3}));
  EXPECT_EQ(context.outputs[0].spec().dtype, DType(DType::kS32));
}

TEST(OpCommonTest, TestPropagateSignatureUnarySuccess) {
  std::vector<xla_serving::data::composite::api::ShapeInferenceValue> inputs;
  inputs.push_back(xla_serving::data::composite::api::HostBufferSpec{
      .dtype = DType(DType::kF32),
      .shape = Shape({4, 5}),
  });

  xla_serving::data::composite::api::ShapeInferenceContext context;
  context.inputs = inputs;

  EXPECT_OK(
      UnaryOpShared::PropagateUnarySignature(context, "TestNegateTransform"));
  ASSERT_EQ(context.outputs.size(), 1);
  EXPECT_TRUE(context.outputs[0].IsSpec());
  EXPECT_EQ(context.outputs[0].spec().shape, Shape({4, 5}));
  EXPECT_EQ(context.outputs[0].spec().dtype, DType(DType::kF32));
}

TEST(OpCommonTest, TestPropagateSignatureRejectsUnknownDimension) {
  const xla_serving::data::composite::api::HostBufferSpec known{
      .dtype = DType(DType::kS32),
      .shape = Shape({2, 3}),
  };
  const xla_serving::data::composite::api::HostBufferSpec unknown{
      .dtype = DType(DType::kS32),
      .shape = PartialShapeOrDie({Dim::Unknown(), Dim::Static(3)}),
  };
  const std::vector<xla_serving::data::composite::api::ShapeInferenceValue>
      unknown_lhs = {unknown, known};
  const std::vector<xla_serving::data::composite::api::ShapeInferenceValue>
      unknown_rhs = {known, unknown};

  xla_serving::data::composite::api::ShapeInferenceContext lhs_context;
  lhs_context.inputs = unknown_lhs;
  EXPECT_THAT(
      BinaryOpShared::PropagateBinarySignature(lhs_context, "TestAddTransform"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("is not fully known")));

  xla_serving::data::composite::api::ShapeInferenceContext rhs_context;
  rhs_context.inputs = unknown_rhs;
  EXPECT_THAT(
      BinaryOpShared::PropagateBinarySignature(rhs_context, "TestAddTransform"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("is not fully known")));
}

TEST(OpCommonTest, TestPropagateSignatureUnaryForwardsUnknownDimension) {
  std::vector<xla_serving::data::composite::api::ShapeInferenceValue> inputs;
  inputs.push_back(xla_serving::data::composite::api::HostBufferSpec{
      .dtype = DType(DType::kF32),
      .shape = PartialShapeOrDie({Dim::Unknown(), Dim::Static(5)}),
  });

  xla_serving::data::composite::api::ShapeInferenceContext context;
  context.inputs = inputs;

  EXPECT_OK(
      UnaryOpShared::PropagateUnarySignature(context, "TestNegateTransform"));
  ASSERT_EQ(context.outputs.size(), 1);
  EXPECT_TRUE(context.outputs[0].IsSpec());
  EXPECT_EQ(context.outputs[0].spec().shape,
            PartialShapeOrDie({Dim::Unknown(), Dim::Static(5)}));
  EXPECT_EQ(context.outputs[0].spec().dtype, DType(DType::kF32));
}

ND_DECLARE_BINARY_OP(FunctorAddTransform, functor::add);
ND_DECLARE_UNARY_OP(FunctorSqrtTransform, functor::sqrt);
ND_DECLARE_BINARY_OP(FunctorMaxTransform, functor::maximum);
ND_DECLARE_BINARY_OP(FunctorBitwiseAndTransform, functor::bitwise_and);
ND_DECLARE_BINARY_OP(FunctorBitwiseOrTransform, functor::bitwise_or);
ND_DECLARE_BINARY_OP(FunctorBitwiseXorTransform, functor::bitwise_xor);
ND_DECLARE_UNARY_OP(FunctorInvertTransform, functor::invert);
ND_DECLARE_BINARY_OP(FunctorLeftShiftTransform, functor::left_shift);
ND_DECLARE_BINARY_OP(FunctorRightShiftTransform, functor::right_shift);

TEST(OpCommonTest, TestStandardFunctorsExecution) {
  // Sqrt
  auto a_sqrt = CreateBuffer<float>({2}, {16.0f, 25.0f});
  ASSERT_OK_AND_ASSIGN(auto sqrt_op, FunctorSqrtTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out_sqrt, (*sqrt_op)(std::move(a_sqrt)));
  EXPECT_EQ(GetData<float>(out_sqrt), std::vector<float>({4.0f, 5.0f}));

  // Add
  auto a_add = CreateBuffer<int32_t>({2}, {10, 20});
  auto b_add = CreateBuffer<int32_t>({2}, {30, 40});
  ASSERT_OK_AND_ASSIGN(auto add_op, FunctorAddTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out_add,
                       (*add_op)(std::move(a_add), std::move(b_add)));
  EXPECT_EQ(GetData<int32_t>(out_add), std::vector<int32_t>({40, 60}));

  // Maximum
  auto a_max = CreateBuffer<int32_t>({2}, {5, 20});
  auto b_max = CreateBuffer<int32_t>({2}, {10, 15});
  ASSERT_OK_AND_ASSIGN(auto max_op, FunctorMaxTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out_max,
                       (*max_op)(std::move(a_max), std::move(b_max)));
  EXPECT_EQ(GetData<int32_t>(out_max), std::vector<int32_t>({10, 20}));

  // Bitwise and shift functors
  auto a_bit = CreateBuffer<int32_t>({2}, {6, -8});
  auto b_bit = CreateBuffer<int32_t>({2}, {3, 2});
  ASSERT_OK_AND_ASSIGN(auto bw_and_op, FunctorBitwiseAndTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto bw_or_op, FunctorBitwiseOrTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto bw_xor_op, FunctorBitwiseXorTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto inv_op, FunctorInvertTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto ls_op, FunctorLeftShiftTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto rs_op, FunctorRightShiftTransform::Create());

  ASSERT_OK_AND_ASSIGN(auto out_bw_and, (*bw_and_op)(a_bit, b_bit));
  ASSERT_OK_AND_ASSIGN(auto out_bw_or, (*bw_or_op)(a_bit, b_bit));
  ASSERT_OK_AND_ASSIGN(auto out_bw_xor, (*bw_xor_op)(a_bit, b_bit));
  ASSERT_OK_AND_ASSIGN(auto out_inv, (*inv_op)(a_bit));
  ASSERT_OK_AND_ASSIGN(auto out_ls, (*ls_op)(a_bit, b_bit));
  ASSERT_OK_AND_ASSIGN(auto out_rs, (*rs_op)(a_bit, b_bit));

  EXPECT_EQ(GetData<int32_t>(out_bw_and), std::vector<int32_t>({2, 0}));
  EXPECT_EQ(GetData<int32_t>(out_bw_or), std::vector<int32_t>({7, -6}));
  EXPECT_EQ(GetData<int32_t>(out_bw_xor), std::vector<int32_t>({5, -6}));
  EXPECT_EQ(GetData<int32_t>(out_inv), std::vector<int32_t>({-7, 7}));
  EXPECT_EQ(GetData<int32_t>(out_ls), std::vector<int32_t>({48, -32}));
  EXPECT_EQ(GetData<int32_t>(out_rs), std::vector<int32_t>({0, -2}));
}

TEST(OpCommonTest, TestApplyUnaryOpTypeMismatchGuard) {
  auto a = CreateBuffer<float>({2}, {1.0f, 2.0f});
  xla_serving::HostBuffer out(a.dtype(), a.shape());

  // Directly invoking ApplyUnaryOp with the WRONG C++ type for the buffer
  EXPECT_THAT(
      ApplyUnaryOp<int32_t>(a, &out, TestNegateFunctor{}, "TestNegate"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Operator 'TestNegate' type mismatch for T and "
                         "in.dtype: F32")));
}

TEST(OpCommonTest, TestApplyBinaryOpTypeMismatchGuard) {
  auto a = CreateBuffer<float>({2}, {1.0f, 2.0f});
  auto b = CreateBuffer<float>({2}, {3.0f, 4.0f});
  xla_serving::HostBuffer out(a.dtype(), a.shape());

  // Directly invoking ApplyBinaryOp with the WRONG C++ type for the buffers
  EXPECT_THAT(
      ApplyBinaryOp<int32_t>(a, b, &out, TestAddFunctor{}, "TestAdd"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Operator 'TestAdd' dtype mismatch: expected S32, but got input "
              "1 as F32, and input 2 as F32, and output as F32")));

  // Directly invoking ApplyBinaryOp with the WRONG dtype for the output buffer
  auto wrong_dtype_out = CreateBuffer<int32_t>({2}, {0, 0});
  EXPECT_THAT(
      ApplyBinaryOp<float>(a, b, &wrong_dtype_out, TestAddFunctor{}, "TestAdd"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Operator 'TestAdd' dtype mismatch: expected F32, but got input "
              "1 as F32, and input 2 as F32, and output as S32")));
}

TEST(OpCommonTest, TestBinaryOpBroadcastScalarVector) {
  auto a_scalar = CreateBuffer<int32_t>({1}, {10});
  auto b_vec = CreateBuffer<int32_t>({3}, {1, 2, 3});
  ASSERT_OK_AND_ASSIGN(auto transform, TestAddTransform::Create());

  // Scalar Left
  ASSERT_OK_AND_ASSIGN(auto out1,
                       (*transform)(std::move(a_scalar), std::move(b_vec)));
  EXPECT_EQ(out1.shape(), Shape({3}));
  EXPECT_EQ(GetData<int32_t>(out1), std::vector<int32_t>({11, 12, 13}));

  // Scalar Right
  auto a_vec = CreateBuffer<int32_t>({3}, {1, 2, 3});
  auto b_scalar = CreateBuffer<int32_t>({1}, {10});
  ASSERT_OK_AND_ASSIGN(auto out2,
                       (*transform)(std::move(a_vec), std::move(b_scalar)));
  EXPECT_EQ(out2.shape(), Shape({3}));
  EXPECT_EQ(GetData<int32_t>(out2), std::vector<int32_t>({11, 12, 13}));
}

TEST(OpCommonTest, TestBinaryOpBroadcast2D) {
  auto a = CreateBuffer<int32_t>({3, 1}, {10, 20, 30});
  auto b = CreateBuffer<int32_t>({1, 4}, {1, 2, 3, 4});
  ASSERT_OK_AND_ASSIGN(auto transform, TestAddTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.shape(), Shape({3, 4}));
  std::vector<int32_t> expected = {
      11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34,
  };
  EXPECT_EQ(GetData<int32_t>(out), expected);
}

TEST(OpCommonTest, TestBinaryOpBroadcastVectorMatrix) {
  auto a = CreateBuffer<int32_t>({4}, {1, 2, 3, 4});
  auto b = CreateBuffer<int32_t>({2, 4}, {10, 20, 30, 40, 50, 60, 70, 80});
  ASSERT_OK_AND_ASSIGN(auto transform, TestAddTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.shape(), Shape({2, 4}));
  std::vector<int32_t> expected = {11, 22, 33, 44, 51, 62, 73, 84};
  EXPECT_EQ(GetData<int32_t>(out), expected);
}

TEST(OpCommonTest, TestBinaryOpBroadcast3D) {
  auto a = CreateBuffer<int32_t>({2, 1, 3}, {1, 2, 3, 4, 5, 6});
  auto b = CreateBuffer<int32_t>({2, 1}, {10, 20});
  ASSERT_OK_AND_ASSIGN(auto transform, TestAddTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.shape(), Shape({2, 2, 3}));
  std::vector<int32_t> expected = {
      11, 12, 13, 21, 22, 23, 14, 15, 16, 24, 25, 26,
  };
  EXPECT_EQ(GetData<int32_t>(out), expected);
}

TEST(OpCommonTest, TestBinaryOpBroadcastSignaturePropagation) {
  std::vector<xla_serving::data::composite::api::ShapeInferenceValue> inputs;
  inputs.push_back(xla_serving::data::composite::api::HostBufferSpec{
      .dtype = DType(DType::kS32),
      .shape = Shape({3, 1}),
  });
  inputs.push_back(xla_serving::data::composite::api::HostBufferSpec{
      .dtype = DType(DType::kS32),
      .shape = Shape({1, 4}),
  });

  xla_serving::data::composite::api::ShapeInferenceContext context;
  context.inputs = inputs;

  EXPECT_OK(
      BinaryOpShared::PropagateBinarySignature(context, "TestAddTransform"));
  ASSERT_EQ(context.outputs.size(), 1);
  EXPECT_TRUE(context.outputs[0].IsSpec());
  EXPECT_EQ(context.outputs[0].spec().shape, Shape({3, 4}));
  EXPECT_EQ(context.outputs[0].spec().dtype, DType(DType::kS32));

  // Incompatible shapes
  std::vector<xla_serving::data::composite::api::ShapeInferenceValue> bad_inputs;
  bad_inputs.push_back(xla_serving::data::composite::api::HostBufferSpec{
      .dtype = DType(DType::kS32),
      .shape = Shape({2, 3}),
  });
  bad_inputs.push_back(xla_serving::data::composite::api::HostBufferSpec{
      .dtype = DType(DType::kS32),
      .shape = Shape({2, 4}),
  });
  xla_serving::data::composite::api::ShapeInferenceContext bad_context;
  bad_context.inputs = bad_inputs;

  EXPECT_THAT(
      BinaryOpShared::PropagateBinarySignature(bad_context, "TestAddTransform"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("TestAddTransform inputs are not broadcast-compatible:"
                         " [2,3] and [2,4]")));
}

TEST(OpCommonTest, TestBinaryOpBroadcastStatusOrFunctor) {
  auto a = CreateBuffer<int32_t>({2, 1}, {10, 20});
  auto b = CreateBuffer<int32_t>({1, 2}, {2, 5});
  ASSERT_OK_AND_ASSIGN(auto transform, TestCheckedDivTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.shape(), Shape({2, 2}));
  std::vector<int32_t> expected = {5, 2, 10, 4};
  EXPECT_EQ(GetData<int32_t>(out), expected);

  // Broadcast with division by zero
  auto a_err = CreateBuffer<int32_t>({2, 1}, {10, 20});
  auto b_err = CreateBuffer<int32_t>({1, 2}, {2, 0});
  EXPECT_THAT((*transform)(std::move(a_err), std::move(b_err)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Division by zero")));
}

TEST(OpCommonTest, TestBinaryOpBroadcast0DScalars) {
  auto a = CreateBuffer<int32_t>({}, {7});
  auto b = CreateBuffer<int32_t>({}, {35});
  ASSERT_OK_AND_ASSIGN(auto transform, TestAddTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.shape(), Shape({}));
  EXPECT_EQ(GetData<int32_t>(out), std::vector<int32_t>({42}));
}

TEST(OpCommonTest, TestBinaryOpBroadcastZeroDimension) {
  auto a = CreateBuffer<float>({2, 0, 5}, {});
  auto b = CreateBuffer<float>({2, 0, 1}, {});
  ASSERT_OK_AND_ASSIGN(auto transform, TestAddTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.shape(), Shape({2, 0, 5}));
  EXPECT_TRUE(GetData<float>(out).empty());
}

TEST(OpCommonTest, TestBinaryOpStringExecution) {
  auto a = CreateBuffer<tsl::tstring>({2, 1}, {"hello ", "good "});
  auto b = CreateBuffer<tsl::tstring>({1, 2}, {"world", "morning"});
  ASSERT_OK_AND_ASSIGN(auto transform, TestAddTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.dtype(), DType(DType::kString));
  EXPECT_EQ(out.shape(), Shape({2, 2}));
  std::vector<tsl::tstring> expected = {"hello world", "hello morning",
                                        "good world", "good morning"};
  EXPECT_EQ(GetData<tsl::tstring>(out), expected);
}

TEST(OpCommonTest, TestBinaryOpBroadcastZeroInnerDimension) {
  auto a = CreateBuffer<float>({2, 3, 0}, {});
  auto b = CreateBuffer<float>({2, 3, 1}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  ASSERT_OK_AND_ASSIGN(auto transform, TestAddTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.shape(), Shape({2, 3, 0}));
  EXPECT_TRUE(GetData<float>(out).empty());
}

TEST(OpCommonTest, TestBinaryOpBroadcastHighRankAlternating) {
  // 5D vs 4D alternating broadcast: [2, 1, 2, 1, 2] + [2, 1, 2, 1] -> [2, 2, 2,
  // 2, 2] None of these dimensions can be coalesced because broadcast
  // directions alternate.
  auto a = CreateBuffer<int32_t>({2, 1, 2, 1, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
  auto b = CreateBuffer<int32_t>({2, 1, 2, 1}, {10, 20, 30, 40});
  ASSERT_OK_AND_ASSIGN(auto transform, TestAddTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.shape(), Shape({2, 2, 2, 2, 2}));
  EXPECT_EQ(out.shape().num_elements(), 32);
  auto data = GetData<int32_t>(out);
  // Spot check first, middle, and last elements:
  // [0,0,0,0,0] (idx 0)  -> a[0,0,0,0,0] (1) + b[0,0,0,0] (10) = 11
  EXPECT_EQ(data.front(), 11);
  // [0,1,0,1,1] (idx 11) -> a[0,0,0,0,1] (2) + b[1,0,1,0] (40) = 42
  EXPECT_EQ(data[11], 42);
  // [1,0,1,0,0] (idx 20) -> a[1,0,1,0,0] (7) + b[0,0,0,0] (10) = 17
  EXPECT_EQ(data[20], 17);
  // [1,0,1,1,0] (idx 22) -> a[1,0,1,0,0] (7) + b[0,0,1,0] (20) = 27
  EXPECT_EQ(data[22], 27);
  // [1,1,1,1,1] (idx 31) -> a[1,0,1,0,1] (8) + b[1,0,1,0] (40) = 48
  EXPECT_EQ(data.back(), 48);
}

TEST(OpCommonTest, TestApplyBinaryOpIncorrectOutputShape) {
  auto in1 = CreateBuffer<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  auto in2 = CreateBuffer<float>({1, 3}, {10.0f, 20.0f, 30.0f});
  // Expected broadcasted output shape is {2, 3}, but allocate {2, 4} instead.
  auto wrong_out = CreateBuffer<float>(
      {2, 4}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});

  // Case 1: Broadcasted inputs with mismatched output shape.
  EXPECT_THAT(ApplyBinaryOp<float>(in1, in2, &wrong_out, functor::add{},
                                   "TestAddTransform"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("output buffer has incorrect shape")));

  // Case 2: Same-shape inputs (fast path) with mismatched output shape.
  EXPECT_THAT(ApplyBinaryOp<float>(in1, in1, &wrong_out, functor::add{},
                                   "TestAddTransform"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("output buffer has incorrect shape")));
}

}  // namespace
}  // namespace xla_serving::data::kernel
