#include "data/kernels/numerical_transforms.h"  // from @xla_serving

#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "Eigen/Core"
#include "data/kernels/math/reduction_ops_common.h"  // from @xla_serving
#include "data/kernels/math/scan_ops.h"  // from @xla_serving
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
using ::xla_serving::data::composite::api::HostBufferSpec;
using ::xla_serving::data::composite::api::ShapeInferenceContext;
using ::xla_serving::data::composite::api::ShapeInferenceValue;
using ::xla_serving::data::test_utils::PartialShapeOrDie;
using ::testing::HasSubstr;
using ::xla::ifrt::DType;
using ::xla::ifrt::Shape;

TEST(NumericalTransformsTest, AddTransformInt32) {
  auto a = CreateBuffer<int32_t>({2}, {1, 2});
  auto b = CreateBuffer<int32_t>({2}, {3, 4});

  ASSERT_OK_AND_ASSIGN(auto transform, AddTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.dtype(), DType(DType::kS32));
  EXPECT_EQ(out.shape(), Shape({2}));
  EXPECT_EQ(GetData<int32_t>(out), std::vector<int32_t>({4, 6}));
}

TEST(NumericalTransformsTest, AddTransformFloat32) {
  auto a = CreateBuffer<float>({2}, {1.5f, 2.5f});
  auto b = CreateBuffer<float>({2}, {3.5f, 4.5f});

  ASSERT_OK_AND_ASSIGN(auto transform, AddTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2}));
  EXPECT_EQ(GetData<float>(out), std::vector<float>({5.0f, 7.0f}));
}

TEST(NumericalTransformsTest, AddTransformInt8AndUInt8) {
  auto a_i8 = CreateBuffer<int8_t>({2}, {-10, 20});
  auto b_i8 = CreateBuffer<int8_t>({2}, {30, -5});
  ASSERT_OK_AND_ASSIGN(auto transform, AddTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out_i8,
                       (*transform)(std::move(a_i8), std::move(b_i8)));
  EXPECT_EQ(out_i8.dtype(), DType(DType::kS8));
  EXPECT_EQ(GetData<int8_t>(out_i8), std::vector<int8_t>({20, 15}));

  auto a_u8 = CreateBuffer<uint8_t>({2}, {100, 200});
  auto b_u8 = CreateBuffer<uint8_t>({2}, {50, 55});
  ASSERT_OK_AND_ASSIGN(auto out_u8,
                       (*transform)(std::move(a_u8), std::move(b_u8)));
  EXPECT_EQ(out_u8.dtype(), DType(DType::kU8));
  EXPECT_EQ(GetData<uint8_t>(out_u8), std::vector<uint8_t>({150, 255}));
}

TEST(NumericalTransformsTest, AddTransformHalfAndBFloat16) {
  auto a_f16 =
      CreateBuffer<Eigen::half>({2}, {Eigen::half(1.5f), Eigen::half(2.5f)});
  auto b_f16 =
      CreateBuffer<Eigen::half>({2}, {Eigen::half(3.5f), Eigen::half(4.5f)});
  ASSERT_OK_AND_ASSIGN(auto transform, AddTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out_f16,
                       (*transform)(std::move(a_f16), std::move(b_f16)));
  EXPECT_EQ(out_f16.dtype(), DType(DType::kF16));
  EXPECT_EQ(GetData<Eigen::half>(out_f16),
            std::vector<Eigen::half>({Eigen::half(5.0f), Eigen::half(7.0f)}));

  auto a_bf16 = CreateBuffer<Eigen::bfloat16>(
      {2}, {Eigen::bfloat16(1.5f), Eigen::bfloat16(2.5f)});
  auto b_bf16 = CreateBuffer<Eigen::bfloat16>(
      {2}, {Eigen::bfloat16(3.5f), Eigen::bfloat16(4.5f)});
  ASSERT_OK_AND_ASSIGN(auto out_bf16,
                       (*transform)(std::move(a_bf16), std::move(b_bf16)));
  EXPECT_EQ(out_bf16.dtype(), DType(DType::kBF16));
  EXPECT_EQ(GetData<Eigen::bfloat16>(out_bf16),
            std::vector<Eigen::bfloat16>(
                {Eigen::bfloat16(5.0f), Eigen::bfloat16(7.0f)}));
}

TEST(NumericalTransformsTest, AddTransformComplex64) {
  using c64 = std::complex<float>;
  auto a = CreateBuffer<c64>({2}, {c64(1.0f, 2.0f), c64(3.0f, 4.0f)});
  auto b = CreateBuffer<c64>({2}, {c64(5.0f, -1.0f), c64(-2.0f, 3.0f)});

  ASSERT_OK_AND_ASSIGN(auto transform, AddTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.dtype(), DType(DType::kC64));
  EXPECT_EQ(GetData<c64>(out),
            std::vector<c64>({c64(6.0f, 1.0f), c64(1.0f, 7.0f)}));
}

TEST(NumericalTransformsTest, AddTransformString) {
  auto a = CreateBuffer<tsl::tstring>({2}, {"foo_", "hello "});
  auto b = CreateBuffer<tsl::tstring>({2}, {"bar", "world"});

  ASSERT_OK_AND_ASSIGN(auto transform, AddTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.dtype(), DType(DType::kString));
  EXPECT_EQ(out.shape(), Shape({2}));
  EXPECT_EQ(GetData<tsl::tstring>(out),
            std::vector<tsl::tstring>({"foo_bar", "hello world"}));
}

TEST(NumericalTransformsTest, SubtractTransformInt32) {
  auto a = CreateBuffer<int32_t>({2}, {5, 8});
  auto b = CreateBuffer<int32_t>({2}, {3, 4});

  ASSERT_OK_AND_ASSIGN(auto transform, SubtractTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.dtype(), DType(DType::kS32));
  EXPECT_EQ(out.shape(), Shape({2}));
  EXPECT_EQ(GetData<int32_t>(out), std::vector<int32_t>({2, 4}));
}

TEST(NumericalTransformsTest, SubtractTransformFloat32) {
  auto a = CreateBuffer<float>({2}, {5.5f, 8.5f});
  auto b = CreateBuffer<float>({2}, {3.0f, 4.5f});

  ASSERT_OK_AND_ASSIGN(auto transform, SubtractTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2}));
  EXPECT_EQ(GetData<float>(out), std::vector<float>({2.5f, 4.0f}));
}

TEST(NumericalTransformsTest, SubtractTransformComplex64) {
  using c64 = std::complex<float>;
  auto a = CreateBuffer<c64>({1}, {c64(5.0f, 3.0f)});
  auto b = CreateBuffer<c64>({1}, {c64(2.0f, 4.0f)});

  ASSERT_OK_AND_ASSIGN(auto transform, SubtractTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.dtype(), DType(DType::kC64));
  EXPECT_EQ(GetData<c64>(out), std::vector<c64>({c64(3.0f, -1.0f)}));
}

TEST(NumericalTransformsTest, MultiplyTransformInt32) {
  auto a = CreateBuffer<int32_t>({2}, {3, 4});
  auto b = CreateBuffer<int32_t>({2}, {5, 6});

  ASSERT_OK_AND_ASSIGN(auto transform, MultiplyTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.dtype(), DType(DType::kS32));
  EXPECT_EQ(out.shape(), Shape({2}));
  EXPECT_EQ(GetData<int32_t>(out), std::vector<int32_t>({15, 24}));
}

TEST(NumericalTransformsTest, MultiplyTransformFloat32) {
  auto a = CreateBuffer<float>({2}, {2.5f, 3.0f});
  auto b = CreateBuffer<float>({2}, {4.0f, 1.5f});

  ASSERT_OK_AND_ASSIGN(auto transform, MultiplyTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2}));
  EXPECT_EQ(GetData<float>(out), std::vector<float>({10.0f, 4.5f}));
}

TEST(NumericalTransformsTest, MultiplyTransformComplex64) {
  using c64 = std::complex<float>;
  auto a = CreateBuffer<c64>({1}, {c64(1.0f, 2.0f)});
  auto b = CreateBuffer<c64>({1}, {c64(3.0f, 4.0f)});

  ASSERT_OK_AND_ASSIGN(auto transform, MultiplyTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.dtype(), DType(DType::kC64));
  EXPECT_EQ(GetData<c64>(out), std::vector<c64>({c64(-5.0f, 10.0f)}));
}

TEST(NumericalTransformsTest, DivideTransformInt32) {
  auto a = CreateBuffer<int32_t>({2}, {12, 20});
  auto b = CreateBuffer<int32_t>({2}, {3, 8});

  ASSERT_OK_AND_ASSIGN(auto transform, DivideTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.dtype(), DType(DType::kS32));
  EXPECT_EQ(out.shape(), Shape({2}));
  EXPECT_EQ(GetData<int32_t>(out), std::vector<int32_t>({4, 2}));
}

TEST(NumericalTransformsTest, DivideTransformFloat32) {
  auto a = CreateBuffer<float>({2}, {10.0f, 15.0f});
  auto b = CreateBuffer<float>({2}, {4.0f, 3.0f});

  ASSERT_OK_AND_ASSIGN(auto transform, DivideTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2}));
  EXPECT_EQ(GetData<float>(out), std::vector<float>({2.5f, 5.0f}));
}

TEST(NumericalTransformsTest, DivideTransformFloat64) {
  auto a = CreateBuffer<double>({2}, {10.0, 15.0});
  auto b = CreateBuffer<double>({2}, {4.0, 3.0});

  ASSERT_OK_AND_ASSIGN(auto transform, DivideTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.dtype(), DType(DType::kF64));
  EXPECT_EQ(out.shape(), Shape({2}));
  EXPECT_EQ(GetData<double>(out), std::vector<double>({2.5, 5.0}));
}

TEST(NumericalTransformsTest, DivideTransformComplex64) {
  using c64 = std::complex<float>;
  auto a = CreateBuffer<c64>({1}, {c64(4.0f, 2.0f)});
  auto b = CreateBuffer<c64>({1}, {c64(2.0f, 0.0f)});

  ASSERT_OK_AND_ASSIGN(auto transform, DivideTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.dtype(), DType(DType::kC64));
  EXPECT_EQ(GetData<c64>(out), std::vector<c64>({c64(2.0f, 1.0f)}));
}

TEST(NumericalTransformsTest, DivideTransformIntegerDivisionByZeroError) {
  auto a = CreateBuffer<int32_t>({2}, {10, 20});
  auto b = CreateBuffer<int32_t>({2}, {2, 0});

  ASSERT_OK_AND_ASSIGN(auto transform, DivideTransform::Create());
  auto status_or = (*transform)(std::move(a), std::move(b));
  EXPECT_FALSE(status_or.ok());
  EXPECT_EQ(status_or.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(status_or.status().message(), "Integer division by zero");
}

TEST(NumericalTransformsTest, DivideTransformIntMinDivideByMinusOne) {
  auto a = CreateBuffer<int32_t>({1}, {std::numeric_limits<int32_t>::min()});
  auto b = CreateBuffer<int32_t>({1}, {-1});

  ASSERT_OK_AND_ASSIGN(auto transform, DivideTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));
  EXPECT_EQ(GetData<int32_t>(out),
            std::vector<int32_t>({std::numeric_limits<int32_t>::min()}));
}

TEST(NumericalTransformsTest, AddTransformBroadcasting) {
  auto a = CreateBuffer<float>({2, 1}, {1.0f, 2.0f});
  auto b = CreateBuffer<float>({1, 3}, {10.0f, 20.0f, 30.0f});

  ASSERT_OK_AND_ASSIGN(auto transform, AddTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2, 3}));
  EXPECT_EQ(GetData<float>(out),
            std::vector<float>({11.0f, 21.0f, 31.0f, 12.0f, 22.0f, 32.0f}));
}

TEST(NumericalTransformsTest, SubtractTransformBroadcasting) {
  auto a = CreateBuffer<float>({2, 1}, {10.0f, 20.0f});
  auto b = CreateBuffer<float>({1, 3}, {1.0f, 2.0f, 3.0f});

  ASSERT_OK_AND_ASSIGN(auto transform, SubtractTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2, 3}));
  EXPECT_EQ(GetData<float>(out),
            std::vector<float>({9.0f, 8.0f, 7.0f, 19.0f, 18.0f, 17.0f}));
}

TEST(NumericalTransformsTest, MultiplyTransformBroadcasting) {
  auto a = CreateBuffer<int32_t>({2, 1}, {2, 3});
  auto b = CreateBuffer<int32_t>({1, 3}, {4, 5, 6});

  ASSERT_OK_AND_ASSIGN(auto transform, MultiplyTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.dtype(), DType(DType::kS32));
  EXPECT_EQ(out.shape(), Shape({2, 3}));
  EXPECT_EQ(GetData<int32_t>(out),
            std::vector<int32_t>({8, 10, 12, 12, 15, 18}));
}

TEST(NumericalTransformsTest, DivideTransformBroadcasting) {
  auto a = CreateBuffer<float>({2, 1}, {12.0f, 24.0f});
  auto b = CreateBuffer<float>({1, 3}, {2.0f, 3.0f, 4.0f});

  ASSERT_OK_AND_ASSIGN(auto transform, DivideTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2, 3}));
  EXPECT_EQ(GetData<float>(out),
            std::vector<float>({6.0f, 4.0f, 3.0f, 12.0f, 8.0f, 6.0f}));
}

TEST(NumericalTransformsTest, DivideTransformDivisionByZeroFloat32) {
  auto a = CreateBuffer<float>({2}, {10.0f, -20.0f});
  auto b = CreateBuffer<float>({2}, {0.0f, 0.0f});

  ASSERT_OK_AND_ASSIGN(auto transform, DivideTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2}));
  auto data = GetData<float>(out);
  EXPECT_TRUE(std::isinf(data[0]) && data[0] > 0);  // +inf
  EXPECT_TRUE(std::isinf(data[1]) && data[1] < 0);  // -inf
}

TEST(NumericalTransformsTest, DivideTransformShapeMismatchError) {
  auto a = CreateBuffer<int32_t>({2}, {1, 2});
  auto b = CreateBuffer<int32_t>({3}, {1, 2, 3});

  ASSERT_OK_AND_ASSIGN(auto transform, DivideTransform::Create());
  auto result = (*transform)(std::move(a), std::move(b));
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(NumericalTransformsTest, DivideTransformToProto) {
  ASSERT_OK_AND_ASSIGN(auto transform, DivideTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto config, transform->ToProto());
  EXPECT_EQ(config.op_name(), "DivideTransform");
}

TEST(NumericalTransformsTest, SqrtTransformFloat32) {
  auto a = CreateBuffer<float>({2, 1}, {4.0f, 16.0f});

  auto transform = SqrtTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2, 1}));
  EXPECT_EQ(GetData<float>(out), std::vector<float>({2.0f, 4.0f}));
}

TEST(NumericalTransformsTest, SqrtTransformFloat64) {
  auto a = CreateBuffer<double>({3}, {9.0, 25.0, 100.0});

  auto transform = SqrtTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kF64));
  EXPECT_EQ(out.shape(), Shape({3}));
  EXPECT_EQ(GetData<double>(out), std::vector<double>({3.0, 5.0, 10.0}));
}

TEST(NumericalTransformsTest, SqrtTransformNegativeFloat32ProducesNaN) {
  auto a = CreateBuffer<float>({2}, {-4.0f, 16.0f});

  auto transform = SqrtTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2}));
  auto data = GetData<float>(out);
  EXPECT_TRUE(std::isnan(data[0]));
  EXPECT_EQ(data[1], 4.0f);
}

TEST(NumericalTransformsTest, SqrtTransformToProto) {
  auto transform = SqrtTransform::Create().value();
  auto config = transform->ToProto().value();
  EXPECT_EQ(config.op_name(), "SqrtTransform");
}

TEST(NumericalTransformsTest, AbsTransformFloat32) {
  auto a = CreateBuffer<float>({2}, {-1.5f, 2.5f});

  auto transform = AbsTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2}));

  auto data = GetData<float>(out);
  EXPECT_EQ(data.size(), 2);
  EXPECT_TRUE(std::abs(data[0] - 1.5f) < 1e-4)
      << "Expected near 1.5f but was " << data[0];
  EXPECT_TRUE(std::abs(data[1] - 2.5f) < 1e-4)
      << "Expected near 2.5f but was " << data[1];
}

TEST(NumericalTransformsTest, AbsTransformToProto) {
  auto transform = AbsTransform::Create().value();
  auto config = transform->ToProto().value();
  EXPECT_EQ(config.op_name(), "AbsTransform");
}

TEST(NumericalTransformsTest, CeilTransformFloat32) {
  auto a = CreateBuffer<float>({2}, {-1.5f, 2.5f});

  auto transform = CeilTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2}));

  auto data = GetData<float>(out);
  EXPECT_EQ(data.size(), 2);
  EXPECT_TRUE(std::abs(data[0] - -1.0f) < 1e-4)
      << "Expected near -1.0f but was " << data[0];
  EXPECT_TRUE(std::abs(data[1] - 3.0f) < 1e-4)
      << "Expected near 3.0f but was " << data[1];
}

TEST(NumericalTransformsTest, CeilTransformToProto) {
  auto transform = CeilTransform::Create().value();
  auto config = transform->ToProto().value();
  EXPECT_EQ(config.op_name(), "CeilTransform");
}

TEST(NumericalTransformsTest, ExpTransformFloat32) {
  auto a = CreateBuffer<float>({2}, {0.0f, 1.0f});

  auto transform = ExpTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2}));

  auto data = GetData<float>(out);
  EXPECT_EQ(data.size(), 2);
  EXPECT_TRUE(std::abs(data[0] - 1.0f) < 1e-4)
      << "Expected near 1.0f but was " << data[0];
  EXPECT_TRUE(std::abs(data[1] - 2.718281828459045f) < 1e-4)
      << "Expected near 2.718281828459045f but was " << data[1];
}

TEST(NumericalTransformsTest, ExpTransformToProto) {
  auto transform = ExpTransform::Create().value();
  auto config = transform->ToProto().value();
  EXPECT_EQ(config.op_name(), "ExpTransform");
}

TEST(NumericalTransformsTest, Expm1TransformFloat32) {
  auto a = CreateBuffer<float>({2}, {0.0f, 1.0f});

  auto transform = Expm1Transform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2}));

  auto data = GetData<float>(out);
  EXPECT_EQ(data.size(), 2);
  EXPECT_TRUE(std::abs(data[0] - 0.0f) < 1e-4)
      << "Expected near 0.0f but was " << data[0];
  EXPECT_TRUE(std::abs(data[1] - 1.718281828459045f) < 1e-4)
      << "Expected near 1.718281828459045f but was " << data[1];
}

TEST(NumericalTransformsTest, Expm1TransformToProto) {
  auto transform = Expm1Transform::Create().value();
  auto config = transform->ToProto().value();
  EXPECT_EQ(config.op_name(), "Expm1Transform");
}

TEST(NumericalTransformsTest, FloorTransformFloat32) {
  auto a = CreateBuffer<float>({2}, {-1.5f, 2.5f});

  auto transform = FloorTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2}));

  auto data = GetData<float>(out);
  EXPECT_EQ(data.size(), 2);
  EXPECT_TRUE(std::abs(data[0] - -2.0f) < 1e-4)
      << "Expected near -2.0f but was " << data[0];
  EXPECT_TRUE(std::abs(data[1] - 2.0f) < 1e-4)
      << "Expected near 2.0f but was " << data[1];
}

TEST(NumericalTransformsTest, FloorTransformToProto) {
  auto transform = FloorTransform::Create().value();
  auto config = transform->ToProto().value();
  EXPECT_EQ(config.op_name(), "FloorTransform");
}

TEST(NumericalTransformsTest, LogTransformFloat32) {
  auto a = CreateBuffer<float>({2}, {1.0f, 2.718281828459045f});

  auto transform = LogTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2}));

  auto data = GetData<float>(out);
  EXPECT_EQ(data.size(), 2);
  EXPECT_TRUE(std::abs(data[0] - 0.0f) < 1e-4)
      << "Expected near 0.0f but was " << data[0];
  EXPECT_TRUE(std::abs(data[1] - 1.0f) < 1e-4)
      << "Expected near 1.0f but was " << data[1];
}

TEST(NumericalTransformsTest, LogTransformToProto) {
  auto transform = LogTransform::Create().value();
  auto config = transform->ToProto().value();
  EXPECT_EQ(config.op_name(), "LogTransform");
}

TEST(NumericalTransformsTest, Log1pTransformFloat32) {
  auto a = CreateBuffer<float>({2}, {0.0f, 1.718281828459045f});

  auto transform = Log1pTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2}));

  auto data = GetData<float>(out);
  EXPECT_EQ(data.size(), 2);
  EXPECT_TRUE(std::abs(data[0] - 0.0f) < 1e-4)
      << "Expected near 0.0f but was " << data[0];
  EXPECT_TRUE(std::abs(data[1] - 1.0f) < 1e-4)
      << "Expected near 1.0f but was " << data[1];
}

TEST(NumericalTransformsTest, Log1pTransformToProto) {
  auto transform = Log1pTransform::Create().value();
  auto config = transform->ToProto().value();
  EXPECT_EQ(config.op_name(), "Log1pTransform");
}

TEST(NumericalTransformsTest, NegativeTransformFloat32) {
  auto a = CreateBuffer<float>({2}, {-1.5f, 2.5f});

  auto transform = NegativeTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2}));

  auto data = GetData<float>(out);
  EXPECT_EQ(data.size(), 2);
  EXPECT_TRUE(std::abs(data[0] - 1.5f) < 1e-4)
      << "Expected near 1.5f but was " << data[0];
  EXPECT_TRUE(std::abs(data[1] - -2.5f) < 1e-4)
      << "Expected near -2.5f but was " << data[1];
}

TEST(NumericalTransformsTest, NegativeTransformToProto) {
  auto transform = NegativeTransform::Create().value();
  auto config = transform->ToProto().value();
  EXPECT_EQ(config.op_name(), "NegativeTransform");
}

TEST(NumericalTransformsTest, ReciprocalTransformFloat32) {
  auto a = CreateBuffer<float>({2}, {2.0f, 4.0f});

  auto transform = ReciprocalTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2}));

  auto data = GetData<float>(out);
  EXPECT_EQ(data.size(), 2);
  EXPECT_TRUE(std::abs(data[0] - 0.5f) < 1e-4)
      << "Expected near 0.5f but was " << data[0];
  EXPECT_TRUE(std::abs(data[1] - 0.25f) < 1e-4)
      << "Expected near 0.25f but was " << data[1];
}

TEST(NumericalTransformsTest, ReciprocalTransformToProto) {
  auto transform = ReciprocalTransform::Create().value();
  auto config = transform->ToProto().value();
  EXPECT_EQ(config.op_name(), "ReciprocalTransform");
}

TEST(NumericalTransformsTest, RintTransformFloat32) {
  auto a = CreateBuffer<float>({2}, {-1.4f, 2.6f});

  auto transform = RintTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2}));

  auto data = GetData<float>(out);
  EXPECT_EQ(data.size(), 2);
  EXPECT_TRUE(std::abs(data[0] - -1.0f) < 1e-4)
      << "Expected near -1.0f but was " << data[0];
  EXPECT_TRUE(std::abs(data[1] - 3.0f) < 1e-4)
      << "Expected near 3.0f but was " << data[1];
}

TEST(NumericalTransformsTest, RintTransformToProto) {
  auto transform = RintTransform::Create().value();
  auto config = transform->ToProto().value();
  EXPECT_EQ(config.op_name(), "RintTransform");
}

TEST(NumericalTransformsTest, RoundTransformFloat32) {
  auto a = CreateBuffer<float>({2}, {-1.4f, 2.6f});

  auto transform = RoundTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2}));

  auto data = GetData<float>(out);
  EXPECT_EQ(data.size(), 2);
  EXPECT_TRUE(std::abs(data[0] - -1.0f) < 1e-4)
      << "Expected near -1.0f but was " << data[0];
  EXPECT_TRUE(std::abs(data[1] - 3.0f) < 1e-4)
      << "Expected near 3.0f but was " << data[1];
}

TEST(NumericalTransformsTest, RoundTransformToProto) {
  auto transform = RoundTransform::Create().value();
  auto config = transform->ToProto().value();
  EXPECT_EQ(config.op_name(), "RoundTransform");
}

TEST(NumericalTransformsTest, RsqrtTransformFloat32) {
  auto a = CreateBuffer<float>({2}, {4.0f, 16.0f});

  auto transform = RsqrtTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2}));

  auto data = GetData<float>(out);
  EXPECT_EQ(data.size(), 2);
  EXPECT_TRUE(std::abs(data[0] - 0.5f) < 1e-4)
      << "Expected near 0.5f but was " << data[0];
  EXPECT_TRUE(std::abs(data[1] - 0.25f) < 1e-4)
      << "Expected near 0.25f but was " << data[1];
}

TEST(NumericalTransformsTest, RsqrtTransformToProto) {
  auto transform = RsqrtTransform::Create().value();
  auto config = transform->ToProto().value();
  EXPECT_EQ(config.op_name(), "RsqrtTransform");
}

TEST(NumericalTransformsTest, SignTransformFloat32) {
  auto a = CreateBuffer<float>({2}, {-2.5f, 3.5f});

  auto transform = SignTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2}));

  auto data = GetData<float>(out);
  EXPECT_EQ(data.size(), 2);
  EXPECT_TRUE(std::abs(data[0] - -1.0f) < 1e-4)
      << "Expected near -1.0f but was " << data[0];
  EXPECT_TRUE(std::abs(data[1] - 1.0f) < 1e-4)
      << "Expected near 1.0f but was " << data[1];
}

TEST(NumericalTransformsTest, SignTransformToProto) {
  auto transform = SignTransform::Create().value();
  auto config = transform->ToProto().value();
  EXPECT_EQ(config.op_name(), "SignTransform");
}

TEST(NumericalTransformsTest, SquareTransformFloat32) {
  auto a = CreateBuffer<float>({2}, {-2.0f, 3.0f});

  auto transform = SquareTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2}));

  auto data = GetData<float>(out);
  EXPECT_EQ(data.size(), 2);
  EXPECT_TRUE(std::abs(data[0] - 4.0f) < 1e-4)
      << "Expected near 4.0f but was " << data[0];
  EXPECT_TRUE(std::abs(data[1] - 9.0f) < 1e-4)
      << "Expected near 9.0f but was " << data[1];
}

TEST(NumericalTransformsTest, SquareTransformToProto) {
  auto transform = SquareTransform::Create().value();
  auto config = transform->ToProto().value();
  EXPECT_EQ(config.op_name(), "SquareTransform");
}
TEST(NumericalTransformsTest, MatMulTransformFloat32) {
  auto a = CreateBuffer<float>({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  auto b = CreateBuffer<float>({2, 2}, {2.0f, 0.0f, 1.0f, 2.0f});

  auto transform = MatMulTransform::Create().value();
  auto out = (*transform)(std::move(a), std::move(b)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2, 2}));

  // [1 2] * [2 0] = [4 4]
  // [3 4]   [1 2]   [10 8]
  EXPECT_EQ(GetData<float>(out), std::vector<float>({4.0f, 4.0f, 10.0f, 8.0f}));
}

TEST(NumericalTransformsTest, MatMulTransformPropagateSignature) {
  auto transform = MatMulTransform::Create().value();
  const std::vector<ShapeInferenceValue> inputs = {
      HostBufferSpec{DType(DType::kF32), Shape({2, 3})},
      HostBufferSpec{DType(DType::kF32), Shape({3, 4})},
  };
  ShapeInferenceContext context;
  context.inputs = inputs;

  ASSERT_OK(transform->PropagateSignature(context));
  ASSERT_EQ(context.outputs.size(), 1);
  EXPECT_EQ(context.outputs[0].spec().shape, Shape({2, 4}));
}

TEST(NumericalTransformsTest, MatMulTransformForwardsUnknownDimensions) {
  auto transform = MatMulTransform::Create().value();
  // The outer dimensions are forwarded; an unknown inner one is accepted.
  const std::vector<ShapeInferenceValue> inputs = {
      HostBufferSpec{DType(DType::kF32),
                     PartialShapeOrDie({Dim::Unknown(), Dim::Unknown()})},
      HostBufferSpec{DType(DType::kF32), Shape({3, 4})},
  };
  ShapeInferenceContext context;
  context.inputs = inputs;
  ASSERT_OK(transform->PropagateSignature(context));
  ASSERT_EQ(context.outputs.size(), 1);
  EXPECT_EQ(context.outputs[0].spec().shape,
            PartialShapeOrDie({Dim::Unknown(), Dim::Static(4)}));

  // Known inner dimensions must match.
  const std::vector<ShapeInferenceValue> mismatched = {
      HostBufferSpec{DType(DType::kF32),
                     PartialShapeOrDie({Dim::Unknown(), Dim::Static(2)})},
      HostBufferSpec{DType(DType::kF32), Shape({3, 4})},
  };
  ShapeInferenceContext mismatched_context;
  mismatched_context.inputs = mismatched;
  EXPECT_THAT(transform->PropagateSignature(mismatched_context),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Inner dimensions must match for MatMul. Got "
                                 "2 and 3")));
}

TEST(NumericalTransformsTest, MatMulTransformRejectsTooLargeOutput) {
  auto transform = MatMulTransform::Create().value();
  constexpr int64_t kBig = int64_t{1} << 40;
  const std::vector<ShapeInferenceValue> inputs = {
      HostBufferSpec{DType(DType::kF32), Shape({kBig, 1})},
      HostBufferSpec{DType(DType::kF32), Shape({1, kBig})},
  };
  ShapeInferenceContext context;
  context.inputs = inputs;

  // The output would have 2**80 entries.
  EXPECT_THAT(
      transform->PropagateSignature(context),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("too large")));
}

TEST(NumericalTransformsTest, LogTransformEdgeCases) {
  auto a = CreateBuffer<float>(
      {3}, {0.0f, -1.0f, std::numeric_limits<float>::infinity()});

  auto transform = LogTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  auto data = GetData<float>(out);
  EXPECT_TRUE(std::isinf(data[0]) && data[0] < 0);  // -inf
  EXPECT_TRUE(std::isnan(data[1]));
  EXPECT_TRUE(std::isinf(data[2]) && data[2] > 0);  // +inf
}

TEST(NumericalTransformsTest, Log1pTransformEdgeCases) {
  auto a = CreateBuffer<float>(
      {3}, {-1.0f, -2.0f, std::numeric_limits<float>::infinity()});

  auto transform = Log1pTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  auto data = GetData<float>(out);
  EXPECT_TRUE(std::isinf(data[0]) && data[0] < 0);  // -inf
  EXPECT_TRUE(std::isnan(data[1]));
  EXPECT_TRUE(std::isinf(data[2]) && data[2] > 0);  // +inf
}

TEST(NumericalTransformsTest, ReciprocalTransformEdgeCases) {
  auto a = CreateBuffer<float>(
      {3}, {0.0f, -0.0f, std::numeric_limits<float>::infinity()});

  auto transform = ReciprocalTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  auto data = GetData<float>(out);
  EXPECT_TRUE(std::isinf(data[0]) && data[0] > 0);  // +inf
  EXPECT_TRUE(std::isinf(data[1]) && data[1] < 0);  // -inf
  EXPECT_EQ(data[2], 0.0f);
}

TEST(NumericalTransformsTest, RsqrtTransformEdgeCases) {
  auto a = CreateBuffer<float>(
      {3}, {0.0f, -1.0f, std::numeric_limits<float>::infinity()});

  auto transform = RsqrtTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  auto data = GetData<float>(out);
  EXPECT_TRUE(std::isinf(data[0]) && data[0] > 0);  // +inf
  EXPECT_TRUE(std::isnan(data[1]));
  EXPECT_EQ(data[2], 0.0f);
}

TEST(NumericalTransformsTest, SignTransformEdgeCases) {
  auto a = CreateBuffer<float>(
      {3}, {0.0f, -0.0f, std::numeric_limits<float>::quiet_NaN()});

  auto transform = SignTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  auto data = GetData<float>(out);
  EXPECT_EQ(data[0], 0.0f);
  EXPECT_TRUE(std::isnan(data[2]));
}

TEST(NumericalTransformsTest, AbsTransformComplex64) {
  auto a =
      CreateBuffer<std::complex<float>>({2}, {{3.0f, 4.0f}, {0.0f, -1.0f}});

  auto transform = AbsTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2}));

  auto data = GetData<float>(out);
  EXPECT_EQ(data.size(), 2);
  EXPECT_NEAR(data[0], 5.0f, 1e-4);
  EXPECT_NEAR(data[1], 1.0f, 1e-4);
}

TEST(NumericalTransformsTest, NegativeTransformComplex64) {
  auto a =
      CreateBuffer<std::complex<float>>({2}, {{1.0f, -2.0f}, {-3.0f, 4.0f}});

  auto transform = NegativeTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kC64));
  EXPECT_EQ(out.shape(), Shape({2}));

  auto data = GetData<std::complex<float>>(out);
  EXPECT_EQ(data[0], std::complex<float>(-1.0f, 2.0f));
  EXPECT_EQ(data[1], std::complex<float>(3.0f, -4.0f));
}

TEST(NumericalTransformsTest, SquareTransformComplex64) {
  auto a = CreateBuffer<std::complex<float>>({2}, {{2.0f, 3.0f}, {1.0f, 1.0f}});

  auto transform = SquareTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kC64));
  EXPECT_EQ(out.shape(), Shape({2}));

  // (2+3i)^2 = 4 - 9 + 12i = -5 + 12i
  // (1+i)^2 = 1 - 1 + 2i = 2i
  auto data = GetData<std::complex<float>>(out);
  EXPECT_NEAR(data[0].real(), -5.0f, 1e-4);
  EXPECT_NEAR(data[0].imag(), 12.0f, 1e-4);
  EXPECT_NEAR(data[1].real(), 0.0f, 1e-4);
  EXPECT_NEAR(data[1].imag(), 2.0f, 1e-4);
}

TEST(NumericalTransformsTest, SqrtTransformComplex64) {
  auto a = CreateBuffer<std::complex<float>>({1}, {{-4.0f, 0.0f}});

  auto transform = SqrtTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kC64));
  EXPECT_EQ(out.shape(), Shape({1}));

  auto data = GetData<std::complex<float>>(out);
  EXPECT_NEAR(data[0].real(), 0.0f, 1e-4);
  EXPECT_NEAR(data[0].imag(), 2.0f, 1e-4);
}

TEST(NumericalTransformsTest, ExpTransformComplex64) {
  auto a = CreateBuffer<std::complex<float>>({1}, {{0.0f, 0.0f}});

  auto transform = ExpTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kC64));
  EXPECT_EQ(out.shape(), Shape({1}));

  auto data = GetData<std::complex<float>>(out);
  EXPECT_NEAR(data[0].real(), 1.0f, 1e-4);
  EXPECT_NEAR(data[0].imag(), 0.0f, 1e-4);
}

TEST(NumericalTransformsTest, ConjTransformComplex64) {
  auto a =
      CreateBuffer<std::complex<float>>({2}, {{1.0f, 2.0f}, {-3.0f, 4.0f}});

  auto transform = ConjTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kC64));
  EXPECT_EQ(out.shape(), Shape({2}));

  auto data = GetData<std::complex<float>>(out);
  EXPECT_NEAR(data[0].real(), 1.0f, 1e-4);
  EXPECT_NEAR(data[0].imag(), -2.0f, 1e-4);
  EXPECT_NEAR(data[1].real(), -3.0f, 1e-4);
  EXPECT_NEAR(data[1].imag(), -4.0f, 1e-4);
}

TEST(NumericalTransformsTest, ComplexAbsTransformComplex64) {
  auto a =
      CreateBuffer<std::complex<float>>({2}, {{3.0f, 4.0f}, {0.0f, -5.0f}});

  auto transform = ComplexAbsTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2}));

  auto data = GetData<float>(out);
  EXPECT_NEAR(data[0], 5.0f, 1e-4);
  EXPECT_NEAR(data[1], 5.0f, 1e-4);
}

TEST(NumericalTransformsTest, AbsTransformHalf) {
  auto a =
      CreateBuffer<Eigen::half>({2}, {Eigen::half(-2.0f), Eigen::half(3.0f)});

  auto transform = AbsTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kF16));
  EXPECT_EQ(out.shape(), Shape({2}));

  auto data = GetData<Eigen::half>(out);
  EXPECT_NEAR(static_cast<float>(data[0]), 2.0f, 1e-3);
  EXPECT_NEAR(static_cast<float>(data[1]), 3.0f, 1e-3);
}

TEST(NumericalTransformsTest, AbsTransformBfloat16) {
  auto a = CreateBuffer<Eigen::bfloat16>(
      {2}, {Eigen::bfloat16(-2.0f), Eigen::bfloat16(3.0f)});

  auto transform = AbsTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kBF16));
  EXPECT_EQ(out.shape(), Shape({2}));

  auto data = GetData<Eigen::bfloat16>(out);
  EXPECT_NEAR(static_cast<float>(data[0]), 2.0f, 1e-2);
  EXPECT_NEAR(static_cast<float>(data[1]), 3.0f, 1e-2);
}

TEST(NumericalTransformsTest, SigmoidTransformFloat32) {
  auto a = CreateBuffer<float>({2}, {0.0f, 2.0f});

  auto transform = SigmoidTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2}));

  auto data = GetData<float>(out);
  EXPECT_NEAR(data[0], 0.5f, 1e-4f);
  EXPECT_NEAR(data[1], 1.0f / (1.0f + std::exp(-2.0f)), 1e-4f);
}

TEST(NumericalTransformsTest, AbsTransformInt8) {
  auto a = CreateBuffer<int8_t>({2}, {-5, 10});

  auto transform = AbsTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kS8));
  EXPECT_EQ(out.shape(), Shape({2}));
  EXPECT_EQ(GetData<int8_t>(out), std::vector<int8_t>({5, 10}));
}

TEST(NumericalTransformsTest, AbsTransformInt16) {
  auto a = CreateBuffer<int16_t>({2}, {-100, 200});

  auto transform = AbsTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kS16));
  EXPECT_EQ(out.shape(), Shape({2}));
  EXPECT_EQ(GetData<int16_t>(out), std::vector<int16_t>({100, 200}));
}

TEST(NumericalTransformsTest, NegativeTransformInt8) {
  auto a = CreateBuffer<int8_t>({2}, {-5, 10});

  auto transform = NegativeTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kS8));
  EXPECT_EQ(out.shape(), Shape({2}));
  EXPECT_EQ(GetData<int8_t>(out), std::vector<int8_t>({5, -10}));
}

TEST(NumericalTransformsTest, NegativeTransformInt16) {
  auto a = CreateBuffer<int16_t>({2}, {-100, 200});

  auto transform = NegativeTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kS16));
  EXPECT_EQ(out.shape(), Shape({2}));
  EXPECT_EQ(GetData<int16_t>(out), std::vector<int16_t>({100, -200}));
}

TEST(NumericalTransformsTest, RoundTransformInt32) {
  auto a = CreateBuffer<int32_t>({2}, {-5, 10});

  auto transform = RoundTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kS32));
  EXPECT_EQ(out.shape(), Shape({2}));
  EXPECT_EQ(GetData<int32_t>(out), std::vector<int32_t>({-5, 10}));
}

TEST(NumericalTransformsTest, RoundTransformInt64) {
  auto a = CreateBuffer<int64_t>({2}, {-100, 200});

  auto transform = RoundTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kS64));
  EXPECT_EQ(out.shape(), Shape({2}));
  EXPECT_EQ(GetData<int64_t>(out), std::vector<int64_t>({-100, 200}));
}

TEST(NumericalTransformsTest, SignTransformInt8) {
  auto a = CreateBuffer<int8_t>({3}, {-5, 0, 10});

  auto transform = SignTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kS8));
  EXPECT_EQ(out.shape(), Shape({3}));
  EXPECT_EQ(GetData<int8_t>(out), std::vector<int8_t>({-1, 0, 1}));
}

TEST(NumericalTransformsTest, SignTransformInt16) {
  auto a = CreateBuffer<int16_t>({3}, {-100, 0, 200});

  auto transform = SignTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kS16));
  EXPECT_EQ(out.shape(), Shape({3}));
  EXPECT_EQ(GetData<int16_t>(out), std::vector<int16_t>({-1, 0, 1}));
}

TEST(NumericalTransformsTest, SquareTransformInt8) {
  auto a = CreateBuffer<int8_t>({2}, {-3, 4});

  auto transform = SquareTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kS8));
  EXPECT_EQ(out.shape(), Shape({2}));
  EXPECT_EQ(GetData<int8_t>(out), std::vector<int8_t>({9, 16}));
}

TEST(NumericalTransformsTest, SquareTransformInt16) {
  auto a = CreateBuffer<int16_t>({2}, {-10, 20});

  auto transform = SquareTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kS16));
  EXPECT_EQ(out.shape(), Shape({2}));
  EXPECT_EQ(GetData<int16_t>(out), std::vector<int16_t>({100, 400}));
}

TEST(NumericalTransformsTest, SquareTransformUint8) {
  auto a = CreateBuffer<uint8_t>({2}, {3, 4});

  auto transform = SquareTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kU8));
  EXPECT_EQ(out.shape(), Shape({2}));
  EXPECT_EQ(GetData<uint8_t>(out), std::vector<uint8_t>({9, 16}));
}

TEST(NumericalTransformsTest, SquareTransformUint16) {
  auto a = CreateBuffer<uint16_t>({2}, {10, 20});

  auto transform = SquareTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kU16));
  EXPECT_EQ(out.shape(), Shape({2}));
  EXPECT_EQ(GetData<uint16_t>(out), std::vector<uint16_t>({100, 400}));
}

TEST(NumericalTransformsTest, SquareTransformUint32) {
  auto a = CreateBuffer<uint32_t>({2}, {10, 20});

  auto transform = SquareTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kU32));
  EXPECT_EQ(out.shape(), Shape({2}));
  EXPECT_EQ(GetData<uint32_t>(out), std::vector<uint32_t>({100, 400}));
}

TEST(NumericalTransformsTest, SquareTransformUint64) {
  auto a = CreateBuffer<uint64_t>({2}, {10, 20});

  auto transform = SquareTransform::Create().value();
  auto out = (*transform)(std::move(a)).value();

  EXPECT_EQ(out.dtype(), DType(DType::kU64));
  EXPECT_EQ(out.shape(), Shape({2}));
  EXPECT_EQ(GetData<uint64_t>(out), std::vector<uint64_t>({100, 400}));
}

template <typename Transform>
absl::StatusOr<xla_serving::HostBuffer> RunTransform(const xla_serving::HostBuffer& input) {
  ASSIGN_OR_RETURN(auto transform, Transform::Create());
  return (*transform)(input);
}

struct TrigFloatTestParam {
  const char* op_name;
  std::function<float(float)> expected_fn;
  absl::StatusOr<xla_serving::HostBuffer> (*execute)(const xla_serving::HostBuffer& input);
  float input_val_0 = 0.0f;
  float input_val_1 = 0.5f;
};

class ParameterizedTrigFloatTest
    : public ::testing::TestWithParam<TrigFloatTestParam> {};

TEST_P(ParameterizedTrigFloatTest, Float32) {
  const auto& param = GetParam();
  auto input = CreateBuffer<float>({2}, {param.input_val_0, param.input_val_1});
  auto out = param.execute(input).value();

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2}));

  auto data = GetData<float>(out);
  EXPECT_NEAR(data[0], param.expected_fn(param.input_val_0), 1e-4);
  EXPECT_NEAR(data[1], param.expected_fn(param.input_val_1), 1e-4);
}

INSTANTIATE_TEST_SUITE_P(
    TrigFloatTests, ParameterizedTrigFloatTest,
    ::testing::Values(
        TrigFloatTestParam{"Sin", [](float x) { return std::sin(x); },
                           RunTransform<SinTransform>},
        TrigFloatTestParam{"Cos", [](float x) { return std::cos(x); },
                           RunTransform<CosTransform>},
        TrigFloatTestParam{"Tan", [](float x) { return std::tan(x); },
                           RunTransform<TanTransform>},
        TrigFloatTestParam{"Asin", [](float x) { return std::asin(x); },
                           RunTransform<AsinTransform>},
        TrigFloatTestParam{"Acos", [](float x) { return std::acos(x); },
                           RunTransform<AcosTransform>},
        TrigFloatTestParam{"Atan", [](float x) { return std::atan(x); },
                           RunTransform<AtanTransform>},
        TrigFloatTestParam{"Sinh", [](float x) { return std::sinh(x); },
                           RunTransform<SinhTransform>},
        TrigFloatTestParam{"Cosh", [](float x) { return std::cosh(x); },
                           RunTransform<CoshTransform>},
        TrigFloatTestParam{"Tanh", [](float x) { return std::tanh(x); },
                           RunTransform<TanhTransform>},
        TrigFloatTestParam{"Asinh", [](float x) { return std::asinh(x); },
                           RunTransform<AsinhTransform>},
        TrigFloatTestParam{"Acosh", [](float x) { return std::acosh(x); },
                           RunTransform<AcoshTransform>, 1.5f, 2.0f},
        TrigFloatTestParam{"Atanh", [](float x) { return std::atanh(x); },
                           RunTransform<AtanhTransform>}),
    [](const ::testing::TestParamInfo<TrigFloatTestParam>& info) {
      return info.param.op_name;
    });

struct TrigOpTestParam {
  const char* op_name;
  absl::StatusOr<xla_serving::HostBuffer> (*execute)(const xla_serving::HostBuffer& input);
};

class ParameterizedTrigIntTest
    : public ::testing::TestWithParam<TrigOpTestParam> {
 protected:
  template <typename T>
  void TestIntSupport() {
    const auto& param = GetParam();
    auto a = CreateBuffer<T>({2}, {0, 1});
    if (std::string(param.op_name) == "Acosh") {
      a = CreateBuffer<T>({2}, {1, 2});
    } else if (std::string(param.op_name) == "Atanh") {
      a = CreateBuffer<T>({2}, {0, 0});
    }

    auto out_or = param.execute(a);
    ASSERT_TRUE(out_or.ok()) << "Op " << param.op_name
                             << " failed for int dtype: " << out_or.status();
    auto out = out_or.value();
    EXPECT_EQ(out.shape(), Shape({2}));
    EXPECT_EQ(GetData<T>(out).size(), 2);
  }
};

TEST_P(ParameterizedTrigIntTest, Int32Support) { TestIntSupport<int32_t>(); }

TEST_P(ParameterizedTrigIntTest, Int64Support) { TestIntSupport<int64_t>(); }

INSTANTIATE_TEST_SUITE_P(
    TrigIntTests, ParameterizedTrigIntTest,
    ::testing::Values(TrigOpTestParam{"Sin", RunTransform<SinTransform>},
                      TrigOpTestParam{"Cos", RunTransform<CosTransform>},
                      TrigOpTestParam{"Tan", RunTransform<TanTransform>},
                      TrigOpTestParam{"Asin", RunTransform<AsinTransform>},
                      TrigOpTestParam{"Acos", RunTransform<AcosTransform>},
                      TrigOpTestParam{"Atan", RunTransform<AtanTransform>},
                      TrigOpTestParam{"Sinh", RunTransform<SinhTransform>},
                      TrigOpTestParam{"Cosh", RunTransform<CoshTransform>},
                      TrigOpTestParam{"Tanh", RunTransform<TanhTransform>},
                      TrigOpTestParam{"Asinh", RunTransform<AsinhTransform>},
                      TrigOpTestParam{"Acosh", RunTransform<AcoshTransform>},
                      TrigOpTestParam{"Atanh", RunTransform<AtanhTransform>}),
    [](const ::testing::TestParamInfo<TrigOpTestParam>& info) {
      return info.param.op_name;
    });

TEST(NumericalTransformsTest, LogicalNotTransformExecution) {
  auto a = CreateBuffer<bool>({2}, {true, false});

  ASSERT_OK_AND_ASSIGN(auto transform, LogicalNotTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a)));

  EXPECT_EQ(out.dtype(), DType(DType::kPred));
  EXPECT_EQ(out.shape(), Shape({2}));
  EXPECT_EQ(GetData<bool>(out), std::vector<bool>({false, true}));
}

TEST(NumericalTransformsTest, LogicalNotTransformToProto) {
  ASSERT_OK_AND_ASSIGN(auto transform, LogicalNotTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto config, transform->ToProto());
  EXPECT_EQ(config.op_name(), "LogicalNotTransform");
}

TEST(NumericalTransformsTest, LogicalAndTransformExecution) {
  auto a = CreateBuffer<bool>({4}, {true, true, false, false});
  auto b = CreateBuffer<bool>({4}, {true, false, true, false});

  ASSERT_OK_AND_ASSIGN(auto transform, LogicalAndTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.dtype(), DType(DType::kPred));
  EXPECT_EQ(out.shape(), Shape({4}));
  EXPECT_EQ(GetData<bool>(out), std::vector<bool>({true, false, false, false}));
}

TEST(NumericalTransformsTest, LogicalAndTransformToProto) {
  ASSERT_OK_AND_ASSIGN(auto transform, LogicalAndTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto config, transform->ToProto());
  EXPECT_EQ(config.op_name(), "LogicalAndTransform");
}

TEST(NumericalTransformsTest, LogicalOrTransformExecution) {
  auto a = CreateBuffer<bool>({4}, {true, true, false, false});
  auto b = CreateBuffer<bool>({4}, {true, false, true, false});

  ASSERT_OK_AND_ASSIGN(auto transform, LogicalOrTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.dtype(), DType(DType::kPred));
  EXPECT_EQ(out.shape(), Shape({4}));
  EXPECT_EQ(GetData<bool>(out), std::vector<bool>({true, true, true, false}));
}

TEST(NumericalTransformsTest, LogicalOrTransformToProto) {
  ASSERT_OK_AND_ASSIGN(auto transform, LogicalOrTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto config, transform->ToProto());
  EXPECT_EQ(config.op_name(), "LogicalOrTransform");
}

TEST(NumericalTransformsTest, LogicalXorTransformExecution) {
  auto a = CreateBuffer<bool>({4}, {true, true, false, false});
  auto b = CreateBuffer<bool>({4}, {true, false, true, false});

  ASSERT_OK_AND_ASSIGN(auto transform, LogicalXorTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.dtype(), DType(DType::kPred));
  EXPECT_EQ(out.shape(), Shape({4}));
  EXPECT_EQ(GetData<bool>(out), std::vector<bool>({false, true, true, false}));
}

TEST(NumericalTransformsTest, LogicalXorTransformToProto) {
  ASSERT_OK_AND_ASSIGN(auto transform, LogicalXorTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto config, transform->ToProto());
  EXPECT_EQ(config.op_name(), "LogicalXorTransform");
}

TEST(NumericalTransformsTest, LogicalBinaryOpShapeMismatch) {
  auto a = CreateBuffer<bool>({2}, {true, false});
  auto b = CreateBuffer<bool>({3}, {true, false, true});

  ASSERT_OK_AND_ASSIGN(auto transform, LogicalAndTransform::Create());
  auto status = (*transform)(std::move(a), std::move(b)).status();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST(NumericalTransformsTest, CumsumTransformBasic) {
  auto a = CreateBuffer<int32_t>({5}, {1, 2, 3, 4, 5});
  auto axis = CreateBuffer<int32_t>({}, {0});

  ASSERT_OK_AND_ASSIGN(auto transform, CumsumTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(a, axis));

  EXPECT_EQ(out.dtype(), DType(DType::kS32));
  EXPECT_EQ(out.shape(), Shape({5}));
  EXPECT_EQ(GetData<int32_t>(out), std::vector<int32_t>({1, 3, 6, 10, 15}));
}

TEST(NumericalTransformsTest, CumsumTransformExclusiveReverse) {
  auto a = CreateBuffer<float>({4}, {1.0f, 2.0f, 3.0f, 4.0f});
  auto axis = CreateBuffer<int32_t>({}, {0});

  // Exclusive
  ASSERT_OK_AND_ASSIGN(auto t_ex, CumsumTransform::Create(/*exclusive=*/true,
                                                          /*reverse=*/false));
  ASSERT_OK_AND_ASSIGN(auto out_ex, (*t_ex)(a, axis));
  EXPECT_EQ(GetData<float>(out_ex),
            std::vector<float>({0.0f, 1.0f, 3.0f, 6.0f}));

  // Reverse
  ASSERT_OK_AND_ASSIGN(auto t_rev, CumsumTransform::Create(/*exclusive=*/false,
                                                           /*reverse=*/true));
  ASSERT_OK_AND_ASSIGN(auto out_rev, (*t_rev)(a, axis));
  EXPECT_EQ(GetData<float>(out_rev),
            std::vector<float>({10.0f, 9.0f, 7.0f, 4.0f}));

  // Exclusive + Reverse
  ASSERT_OK_AND_ASSIGN(auto t_both, CumsumTransform::Create(/*exclusive=*/true,
                                                            /*reverse=*/true));
  ASSERT_OK_AND_ASSIGN(auto out_both, (*t_both)(a, axis));
  EXPECT_EQ(GetData<float>(out_both),
            std::vector<float>({9.0f, 7.0f, 4.0f, 0.0f}));
}

TEST(NumericalTransformsTest, CumsumTransform2D) {
  auto a = CreateBuffer<int32_t>({2, 3}, {1, 2, 3, 4, 5, 6});

  // Axis 0
  auto axis0 = CreateBuffer<int32_t>({}, {0});
  ASSERT_OK_AND_ASSIGN(auto t, CumsumTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out0, (*t)(a, axis0));
  EXPECT_EQ(GetData<int32_t>(out0), std::vector<int32_t>({1, 2, 3, 5, 7, 9}));

  // Axis 1
  auto axis1 = CreateBuffer<int32_t>({}, {1});
  ASSERT_OK_AND_ASSIGN(auto out1, (*t)(a, axis1));
  EXPECT_EQ(GetData<int32_t>(out1), std::vector<int32_t>({1, 3, 6, 4, 9, 15}));

  // Negative axis -1 (equivalent to axis 1)
  auto axis_neg = CreateBuffer<int64_t>({}, {-1});
  ASSERT_OK_AND_ASSIGN(auto out_neg, (*t)(a, axis_neg));
  EXPECT_EQ(GetData<int32_t>(out_neg),
            std::vector<int32_t>({1, 3, 6, 4, 9, 15}));
}

TEST(NumericalTransformsTest, CumsumTransformEmpty) {
  auto a = CreateBuffer<float>({0}, {});
  auto axis = CreateBuffer<int32_t>({}, {0});

  ASSERT_OK_AND_ASSIGN(auto t, CumsumTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*t)(a, axis));
  EXPECT_EQ(out.shape(), Shape({0}));
  EXPECT_EQ(GetData<float>(out).size(), 0);
}

TEST(NumericalTransformsTest, CumsumTransformInvalidAxis) {
  auto a = CreateBuffer<float>({2, 3}, {1, 2, 3, 4, 5, 6});

  // Non-scalar axis
  auto axis_non_scalar = CreateBuffer<int32_t>({1}, {0});
  ASSERT_OK_AND_ASSIGN(auto t, CumsumTransform::Create());
  auto status_non_scalar = (*t)(a, axis_non_scalar).status();
  EXPECT_FALSE(status_non_scalar.ok());
  EXPECT_THAT(status_non_scalar.message(),
              testing::HasSubstr("axis must be a scalar"));

  // Out of bounds axis
  auto axis_oob = CreateBuffer<int32_t>({}, {2});
  auto status_oob = (*t)(a, axis_oob).status();
  EXPECT_FALSE(status_oob.ok());
  EXPECT_THAT(status_oob.message(),
              testing::HasSubstr("Expected scan axis in the range [-2, 2)"));
}

TEST(NumericalTransformsTest, CumprodTransformBasic) {
  auto a = CreateBuffer<int32_t>({5}, {1, 2, 3, 4, 5});
  auto axis = CreateBuffer<int32_t>({}, {0});

  ASSERT_OK_AND_ASSIGN(auto transform, CumprodTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(a, axis));

  EXPECT_EQ(out.dtype(), DType(DType::kS32));
  EXPECT_EQ(out.shape(), Shape({5}));
  EXPECT_EQ(GetData<int32_t>(out), std::vector<int32_t>({1, 2, 6, 24, 120}));
}

TEST(NumericalTransformsTest, CumprodTransformExclusiveReverse) {
  auto a = CreateBuffer<float>({4}, {1.0f, 2.0f, 3.0f, 4.0f});
  auto axis = CreateBuffer<int32_t>({}, {0});

  // Exclusive (init is 1.0f)
  ASSERT_OK_AND_ASSIGN(auto t_ex, CumprodTransform::Create(/*exclusive=*/true,
                                                           /*reverse=*/false));
  ASSERT_OK_AND_ASSIGN(auto out_ex, (*t_ex)(a, axis));
  EXPECT_EQ(GetData<float>(out_ex),
            std::vector<float>({1.0f, 1.0f, 2.0f, 6.0f}));

  // Reverse
  ASSERT_OK_AND_ASSIGN(auto t_rev, CumprodTransform::Create(/*exclusive=*/false,
                                                            /*reverse=*/true));
  ASSERT_OK_AND_ASSIGN(auto out_rev, (*t_rev)(a, axis));
  EXPECT_EQ(GetData<float>(out_rev),
            std::vector<float>({24.0f, 24.0f, 12.0f, 4.0f}));
}

TEST(NumericalTransformsTest, CumulativeLogsumexpTransformBasic) {
  auto a = CreateBuffer<float>({3}, {1.0f, 2.0f, 3.0f});
  auto axis = CreateBuffer<int32_t>({}, {0});

  ASSERT_OK_AND_ASSIGN(auto transform, CumulativeLogsumexpTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(a, axis));

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({3}));
  auto data = GetData<float>(out);
  EXPECT_NEAR(data[0], 1.0f, 1e-4f);
  EXPECT_NEAR(data[1], std::log(std::exp(1.0f) + std::exp(2.0f)), 1e-4f);
  EXPECT_NEAR(data[2],
              std::log(std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f)),
              1e-4f);
}

TEST(NumericalTransformsTest, CumulativeLogsumexpTransformExclusive) {
  auto a = CreateBuffer<float>({3}, {1.0f, 2.0f, 3.0f});
  auto axis = CreateBuffer<int32_t>({}, {0});

  ASSERT_OK_AND_ASSIGN(auto transform,
                       CumulativeLogsumexpTransform::Create(/*exclusive=*/true,
                                                            /*reverse=*/false));
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(a, axis));

  auto data = GetData<float>(out);
  EXPECT_TRUE(std::isinf(data[0]) && data[0] < 0);
  EXPECT_NEAR(data[1], 1.0f, 1e-4f);
  EXPECT_NEAR(data[2], std::log(std::exp(1.0f) + std::exp(2.0f)), 1e-4f);
}

TEST(NumericalTransformsTest, CumulativeLogsumexpTransformStability) {
  auto a = CreateBuffer<float>({2}, {1000.0f, 1000.0f});
  auto axis = CreateBuffer<int32_t>({}, {0});

  ASSERT_OK_AND_ASSIGN(auto transform, CumulativeLogsumexpTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(a, axis));

  auto data = GetData<float>(out);
  EXPECT_NEAR(data[1], 1000.0f + std::log(2.0f), 1e-3f);
}

TEST(NumericalTransformsTest, InvertTransformInt32) {
  auto a = CreateBuffer<int32_t>({3}, {0, -1, 1});
  ASSERT_OK_AND_ASSIGN(auto transform, InvertTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a)));
  EXPECT_EQ(out.dtype(), DType(DType::kS32));
  EXPECT_EQ(out.shape(), Shape({3}));
  EXPECT_EQ(GetData<int32_t>(out), std::vector<int32_t>({-1, 0, -2}));
}

TEST(NumericalTransformsTest, InvertTransformUint8) {
  auto a = CreateBuffer<uint8_t>({3}, {0, 255, 1});
  ASSERT_OK_AND_ASSIGN(auto transform, InvertTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a)));
  EXPECT_EQ(out.dtype(), DType(DType::kU8));
  EXPECT_EQ(out.shape(), Shape({3}));
  EXPECT_EQ(GetData<uint8_t>(out), std::vector<uint8_t>({255, 0, 254}));
}

TEST(NumericalTransformsTest, BitwiseBinaryTransformsInt32AndUint8) {
  auto lhs_i32 = CreateBuffer<int32_t>({4}, {0, 5, 3, 14});
  auto rhs_i32 = CreateBuffer<int32_t>({4}, {5, 0, 7, 11});

  ASSERT_OK_AND_ASSIGN(auto and_t, BitwiseAndTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto or_t, BitwiseOrTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto xor_t, BitwiseXorTransform::Create());

  ASSERT_OK_AND_ASSIGN(auto and_out, (*and_t)(lhs_i32, rhs_i32));
  ASSERT_OK_AND_ASSIGN(auto or_out, (*or_t)(lhs_i32, rhs_i32));
  ASSERT_OK_AND_ASSIGN(auto xor_out, (*xor_t)(lhs_i32, rhs_i32));

  EXPECT_EQ(and_out.dtype(), DType(DType::kS32));
  EXPECT_EQ(GetData<int32_t>(and_out), std::vector<int32_t>({0, 0, 3, 10}));
  EXPECT_EQ(or_out.dtype(), DType(DType::kS32));
  EXPECT_EQ(GetData<int32_t>(or_out), std::vector<int32_t>({5, 5, 7, 15}));
  EXPECT_EQ(xor_out.dtype(), DType(DType::kS32));
  EXPECT_EQ(GetData<int32_t>(xor_out), std::vector<int32_t>({5, 5, 4, 5}));

  auto lhs_u8 = CreateBuffer<uint8_t>({3}, {0xFF, 0x0F, 0xAA});
  auto rhs_u8 = CreateBuffer<uint8_t>({3}, {0x0F, 0xF0, 0x55});
  ASSERT_OK_AND_ASSIGN(auto and_u8, (*and_t)(lhs_u8, rhs_u8));
  ASSERT_OK_AND_ASSIGN(auto or_u8, (*or_t)(lhs_u8, rhs_u8));
  ASSERT_OK_AND_ASSIGN(auto xor_u8, (*xor_t)(lhs_u8, rhs_u8));
  EXPECT_EQ(GetData<uint8_t>(and_u8), std::vector<uint8_t>({0x0F, 0x00, 0x00}));
  EXPECT_EQ(GetData<uint8_t>(or_u8), std::vector<uint8_t>({0xFF, 0xFF, 0xFF}));
  EXPECT_EQ(GetData<uint8_t>(xor_u8), std::vector<uint8_t>({0xF0, 0xFF, 0xFF}));
}

TEST(NumericalTransformsTest, ShiftTransformsSignedAndUnsignedWithClamping) {
  ASSERT_OK_AND_ASSIGN(auto ls_t, LeftShiftTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto rs_t, RightShiftTransform::Create());

  // Positive and negative LHS for int32_t
  auto lhs_i32 = CreateBuffer<int32_t>({4}, {-1, -5, 3, 14});
  auto rhs_i32 = CreateBuffer<int32_t>({4}, {5, 0, 7, 3});
  ASSERT_OK_AND_ASSIGN(auto ls_i32, (*ls_t)(lhs_i32, rhs_i32));
  ASSERT_OK_AND_ASSIGN(auto rs_i32, (*rs_t)(lhs_i32, rhs_i32));
  EXPECT_EQ(GetData<int32_t>(ls_i32),
            std::vector<int32_t>({-32, -5, 384, 112}));
  EXPECT_EQ(GetData<int32_t>(rs_i32), std::vector<int32_t>({-1, -5, 0, 1}));

  // Clamping out-of-range shift counts (negative -> 0, >= bitwidth -> bitwidth
  // - 1)
  auto lhs_clamp = CreateBuffer<int8_t>({3}, {-16, 1, -1});
  auto rhs_clamp = CreateBuffer<int8_t>({3}, {-2, 64, 100});
  ASSERT_OK_AND_ASSIGN(auto ls_clamp, (*ls_t)(lhs_clamp, rhs_clamp));
  ASSERT_OK_AND_ASSIGN(auto rs_clamp, (*rs_t)(lhs_clamp, rhs_clamp));
  // int8_t bitwidth - 1 is 7:
  // -16 << 0 = -16, 1 << 7 = -128, -1 << 7 = -128
  EXPECT_EQ(GetData<int8_t>(ls_clamp), std::vector<int8_t>({-16, -128, -128}));
  // -16 >> 0 = -16, 1 >> 7 = 0, -1 >> 7 = -1
  EXPECT_EQ(GetData<int8_t>(rs_clamp), std::vector<int8_t>({-16, 0, -1}));

  // Unsigned logical right shift
  auto lhs_u32 = CreateBuffer<uint32_t>({2}, {0x80000000u, 0xFF000000u});
  auto rhs_u32 = CreateBuffer<uint32_t>({2}, {1u, 31u});
  ASSERT_OK_AND_ASSIGN(auto rs_u32, (*rs_t)(lhs_u32, rhs_u32));
  EXPECT_EQ(GetData<uint32_t>(rs_u32),
            std::vector<uint32_t>({0x40000000u, 1u}));
}

TEST(NumericalTransformsTest, BitwiseAndShiftTransformsBroadcastingAndToProto) {
  auto lhs = CreateBuffer<int16_t>({2, 1}, {3, 5});
  auto rhs = CreateBuffer<int16_t>({1, 2}, {1, 2});

  ASSERT_OK_AND_ASSIGN(auto and_t, BitwiseAndTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto or_t, BitwiseOrTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto xor_t, BitwiseXorTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto inv_t, InvertTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto ls_t, LeftShiftTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto rs_t, RightShiftTransform::Create());

  ASSERT_OK_AND_ASSIGN(auto and_out, (*and_t)(lhs, rhs));
  EXPECT_EQ(and_out.shape(), Shape({2, 2}));
  EXPECT_EQ(GetData<int16_t>(and_out), std::vector<int16_t>({1, 2, 1, 0}));

  ASSERT_OK_AND_ASSIGN(auto ls_out, (*ls_t)(lhs, rhs));
  EXPECT_EQ(ls_out.shape(), Shape({2, 2}));
  EXPECT_EQ(GetData<int16_t>(ls_out), std::vector<int16_t>({6, 12, 10, 20}));

  ASSERT_OK_AND_ASSIGN(auto and_cfg, and_t->ToProto());
  EXPECT_EQ(and_cfg.op_name(), "BitwiseAndTransform");
  ASSERT_OK_AND_ASSIGN(auto or_cfg, or_t->ToProto());
  EXPECT_EQ(or_cfg.op_name(), "BitwiseOrTransform");
  ASSERT_OK_AND_ASSIGN(auto xor_cfg, xor_t->ToProto());
  EXPECT_EQ(xor_cfg.op_name(), "BitwiseXorTransform");
  ASSERT_OK_AND_ASSIGN(auto inv_cfg, inv_t->ToProto());
  EXPECT_EQ(inv_cfg.op_name(), "InvertTransform");
  ASSERT_OK_AND_ASSIGN(auto ls_cfg, ls_t->ToProto());
  EXPECT_EQ(ls_cfg.op_name(), "LeftShiftTransform");
  ASSERT_OK_AND_ASSIGN(auto rs_cfg, rs_t->ToProto());
  EXPECT_EQ(rs_cfg.op_name(), "RightShiftTransform");

  // Unsupported dtype (float) returns kUnimplemented
  auto f_lhs = CreateBuffer<float>({2}, {1.0f, 2.0f});
  auto f_rhs = CreateBuffer<float>({2}, {1.0f, 2.0f});
  EXPECT_EQ((*and_t)(f_lhs, f_rhs).status().code(),
            absl::StatusCode::kUnimplemented);
  EXPECT_EQ((*or_t)(f_lhs, f_rhs).status().code(),
            absl::StatusCode::kUnimplemented);
  EXPECT_EQ((*xor_t)(f_lhs, f_rhs).status().code(),
            absl::StatusCode::kUnimplemented);
  EXPECT_EQ((*inv_t)(f_lhs).status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_EQ((*ls_t)(f_lhs, f_rhs).status().code(),
            absl::StatusCode::kUnimplemented);
  EXPECT_EQ((*rs_t)(f_lhs, f_rhs).status().code(),
            absl::StatusCode::kUnimplemented);
}

TEST(NumericalTransformsTest, IsFiniteTransformFloat) {
  auto a = CreateBuffer<float>(
      {5}, {1.0f, -2.5f, std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN()});
  ASSERT_OK_AND_ASSIGN(auto transform, IsFiniteTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a)));
  EXPECT_EQ(out.dtype(), DType(DType::kPred));
  EXPECT_EQ(out.shape(), Shape({5}));
  EXPECT_EQ(GetData<bool>(out),
            std::vector<bool>({true, true, false, false, false}));
}

TEST(NumericalTransformsTest, IsInfTransformFloat) {
  auto a =
      CreateBuffer<float>({4}, {1.0f, std::numeric_limits<float>::infinity(),
                                -std::numeric_limits<float>::infinity(),
                                std::numeric_limits<float>::quiet_NaN()});
  ASSERT_OK_AND_ASSIGN(auto transform, IsInfTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a)));
  EXPECT_EQ(out.dtype(), DType(DType::kPred));
  EXPECT_EQ(out.shape(), Shape({4}));
  EXPECT_EQ(GetData<bool>(out), std::vector<bool>({false, true, true, false}));
}

TEST(NumericalTransformsTest, IsNanTransformFloat) {
  auto a =
      CreateBuffer<float>({3}, {1.0f, std::numeric_limits<float>::infinity(),
                                std::numeric_limits<float>::quiet_NaN()});
  ASSERT_OK_AND_ASSIGN(auto transform, IsNanTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a)));
  EXPECT_EQ(out.dtype(), DType(DType::kPred));
  EXPECT_EQ(out.shape(), Shape({3}));
  EXPECT_EQ(GetData<bool>(out), std::vector<bool>({false, false, true}));
}

TEST(NumericalTransformsTest, DawsnTransformFloat) {
  auto a = CreateBuffer<float>({2}, {0.0f, 1.0f});
  ASSERT_OK_AND_ASSIGN(auto transform, DawsnTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a)));
  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2}));
  auto data = GetData<float>(out);
  EXPECT_NEAR(data[0], 0.0f, 1e-5);
  EXPECT_NEAR(data[1], 0.5380795f, 1e-5);
}

TEST(NumericalTransformsTest, DigammaTransformFloat) {
  auto a = CreateBuffer<float>({2}, {1.0f, 2.0f});
  ASSERT_OK_AND_ASSIGN(auto transform, DigammaTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a)));
  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2}));
  auto data = GetData<float>(out);
  EXPECT_NEAR(data[0], -0.57721566f, 1e-4);
  EXPECT_NEAR(data[1], 0.42278433f, 1e-4);
}

TEST(NumericalTransformsTest, ErfTransformFloat) {
  auto a = CreateBuffer<float>({2}, {0.0f, 1.0f});
  ASSERT_OK_AND_ASSIGN(auto transform, ErfTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a)));
  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2}));
  auto data = GetData<float>(out);
  EXPECT_NEAR(data[0], std::erf(0.0f), 1e-5);
  EXPECT_NEAR(data[1], std::erf(1.0f), 1e-5);
}

TEST(NumericalTransformsTest, ErfcTransformFloat) {
  auto a = CreateBuffer<float>({2}, {0.0f, 1.0f});
  ASSERT_OK_AND_ASSIGN(auto transform, ErfcTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a)));
  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2}));
  auto data = GetData<float>(out);
  EXPECT_NEAR(data[0], std::erfc(0.0f), 1e-5);
  EXPECT_NEAR(data[1], std::erfc(1.0f), 1e-5);
}

TEST(NumericalTransformsTest, ErfinvTransformFloat) {
  auto a = CreateBuffer<float>({2}, {0.0f, 0.5f});
  ASSERT_OK_AND_ASSIGN(auto transform, ErfinvTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a)));
  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2}));
  auto data = GetData<float>(out);
  EXPECT_NEAR(data[0], 0.0f, 1e-5);
  EXPECT_NEAR(data[1], 0.47693627f, 1e-4);
}

TEST(NumericalTransformsTest, ErfinvTransformDouble) {
  auto a = CreateBuffer<double>({2}, {0.0, 0.5});
  ASSERT_OK_AND_ASSIGN(auto transform, ErfinvTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a)));
  EXPECT_EQ(out.dtype(), DType(DType::kF64));
  EXPECT_EQ(out.shape(), Shape({2}));
  auto data = GetData<double>(out);
  EXPECT_NEAR(data[0], 0.0, 1e-5);
  EXPECT_NEAR(data[1], 0.47693627, 1e-4);
}

TEST(NumericalTransformsTest, ExpintTransformFloat) {
  auto a = CreateBuffer<float>({2}, {1.0f, 2.0f});
  ASSERT_OK_AND_ASSIGN(auto transform, ExpintTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a)));
  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2}));
  auto data = GetData<float>(out);
  EXPECT_NEAR(data[0], 1.8951178f, 1e-4);
  EXPECT_NEAR(data[1], 4.954234f, 1e-4);
}

TEST(NumericalTransformsTest, FresnelCosAndSinTransformFloat) {
  auto a = CreateBuffer<float>({2}, {0.0f, 1.0f});
  ASSERT_OK_AND_ASSIGN(auto cos_transform, FresnelCosTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out_cos, (*cos_transform)(a));
  EXPECT_EQ(out_cos.dtype(), DType(DType::kF32));
  auto cos_data = GetData<float>(out_cos);
  EXPECT_NEAR(cos_data[0], 0.0f, 1e-5);
  EXPECT_NEAR(cos_data[1], 0.7798934f, 1e-4);

  ASSERT_OK_AND_ASSIGN(auto sin_transform, FresnelSinTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out_sin, (*sin_transform)(std::move(a)));
  EXPECT_EQ(out_sin.dtype(), DType(DType::kF32));
  auto sin_data = GetData<float>(out_sin);
  EXPECT_NEAR(sin_data[0], 0.0f, 1e-5);
  EXPECT_NEAR(sin_data[1], 0.4382591f, 1e-4);
}

TEST(NumericalTransformsTest, LgammaTransformFloat) {
  auto a = CreateBuffer<float>({3}, {1.0f, 2.0f, 3.0f});
  ASSERT_OK_AND_ASSIGN(auto transform, LgammaTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a)));
  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({3}));
  auto data = GetData<float>(out);
  EXPECT_NEAR(data[0], std::lgamma(1.0f), 1e-5);
  EXPECT_NEAR(data[1], std::lgamma(2.0f), 1e-5);
  EXPECT_NEAR(data[2], std::lgamma(3.0f), 1e-5);
}

TEST(NumericalTransformsTest, NdtriTransformFloat) {
  auto a = CreateBuffer<float>({2}, {0.5f, 0.8413447f});
  ASSERT_OK_AND_ASSIGN(auto transform, NdtriTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a)));
  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2}));
  auto data = GetData<float>(out);
  EXPECT_NEAR(data[0], 0.0f, 1e-5);
  EXPECT_NEAR(data[1], 1.0f, 1e-4);
}

TEST(NumericalTransformsTest, SpenceTransformFloat) {
  auto a = CreateBuffer<float>({2}, {1.0f, 2.0f});
  ASSERT_OK_AND_ASSIGN(auto transform, SpenceTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a)));
  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2}));
  auto data = GetData<float>(out);
  EXPECT_NEAR(data[0], 0.0f, 1e-5);
  EXPECT_NEAR(data[1], -0.822467f, 1e-4);
}

TEST(NumericalTransformsTest, BesselTransformsFloat) {
  auto zero = CreateBuffer<float>({1}, {0.0f});
  auto one = CreateBuffer<float>({1}, {1.0f});

  // BesselI0(0) = 1, BesselI0e(0) = 1
  ASSERT_OK_AND_ASSIGN(auto bessel_i0, BesselI0Transform::Create());
  ASSERT_OK_AND_ASSIGN(auto out_i0, (*bessel_i0)(zero));
  EXPECT_NEAR(GetData<float>(out_i0)[0], 1.0f, 1e-4);

  ASSERT_OK_AND_ASSIGN(auto bessel_i0e, BesselI0eTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out_i0e, (*bessel_i0e)(zero));
  EXPECT_NEAR(GetData<float>(out_i0e)[0], 1.0f, 1e-4);

  // BesselI1(0) = 0, BesselI1e(0) = 0
  ASSERT_OK_AND_ASSIGN(auto bessel_i1, BesselI1Transform::Create());
  ASSERT_OK_AND_ASSIGN(auto out_i1, (*bessel_i1)(zero));
  EXPECT_NEAR(GetData<float>(out_i1)[0], 0.0f, 1e-4);

  ASSERT_OK_AND_ASSIGN(auto bessel_i1e, BesselI1eTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out_i1e, (*bessel_i1e)(zero));
  EXPECT_NEAR(GetData<float>(out_i1e)[0], 0.0f, 1e-4);

  // BesselJ0(0) = 1, BesselJ1(0) = 0
  ASSERT_OK_AND_ASSIGN(auto bessel_j0, BesselJ0Transform::Create());
  ASSERT_OK_AND_ASSIGN(auto out_j0, (*bessel_j0)(zero));
  EXPECT_NEAR(GetData<float>(out_j0)[0], 1.0f, 1e-4);

  ASSERT_OK_AND_ASSIGN(auto bessel_j1, BesselJ1Transform::Create());
  ASSERT_OK_AND_ASSIGN(auto out_j1, (*bessel_j1)(zero));
  EXPECT_NEAR(GetData<float>(out_j1)[0], 0.0f, 1e-4);

  // BesselK0(1) ~= 0.421024f, BesselK0e(1) ~= 1.14446f
  ASSERT_OK_AND_ASSIGN(auto bessel_k0, BesselK0Transform::Create());
  ASSERT_OK_AND_ASSIGN(auto out_k0, (*bessel_k0)(one));
  EXPECT_NEAR(GetData<float>(out_k0)[0], 0.421024f, 1e-4);

  ASSERT_OK_AND_ASSIGN(auto bessel_k0e, BesselK0eTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out_k0e, (*bessel_k0e)(one));
  EXPECT_NEAR(GetData<float>(out_k0e)[0], 1.14446f, 1e-4);

  // BesselK1(1) ~= 0.601907f, BesselK1e(1) ~= 1.63615f
  ASSERT_OK_AND_ASSIGN(auto bessel_k1, BesselK1Transform::Create());
  ASSERT_OK_AND_ASSIGN(auto out_k1, (*bessel_k1)(one));
  EXPECT_NEAR(GetData<float>(out_k1)[0], 0.601907f, 1e-4);

  ASSERT_OK_AND_ASSIGN(auto bessel_k1e, BesselK1eTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out_k1e, (*bessel_k1e)(one));
  EXPECT_NEAR(GetData<float>(out_k1e)[0], 1.63615f, 1e-4);

  // BesselY0(1) ~= 0.088257f, BesselY1(1) ~= -0.781212f
  ASSERT_OK_AND_ASSIGN(auto bessel_y0, BesselY0Transform::Create());
  ASSERT_OK_AND_ASSIGN(auto out_y0, (*bessel_y0)(one));
  EXPECT_NEAR(GetData<float>(out_y0)[0], 0.088257f, 1e-4);

  ASSERT_OK_AND_ASSIGN(auto bessel_y1, BesselY1Transform::Create());
  ASSERT_OK_AND_ASSIGN(auto out_y1, (*bessel_y1)(one));
  EXPECT_NEAR(GetData<float>(out_y1)[0], -0.781212f, 1e-4);
}

TEST(NumericalTransformsTest, UnsupportedDTypeReturnsError) {
  auto b = CreateBuffer<bool>({2}, {true, false});
  ASSERT_OK_AND_ASSIGN(auto sin_transform, SinTransform::Create());
  auto res = (*sin_transform)(std::move(b));
  EXPECT_FALSE(res.ok());
  EXPECT_EQ(res.status().code(), absl::StatusCode::kUnimplemented);

  auto c = CreateBuffer<std::complex<float>>({2}, {{1.0f, 2.0f}, {3.0f, 4.0f}});
  ASSERT_OK_AND_ASSIGN(auto ceil_transform, CeilTransform::Create());
  auto ceil_res = (*ceil_transform)(std::move(c));
  EXPECT_FALSE(ceil_res.ok());
  EXPECT_EQ(ceil_res.status().code(), absl::StatusCode::kUnimplemented);

  auto int_buf = CreateBuffer<int32_t>({2}, {1, 2});
  ASSERT_OK_AND_ASSIGN(auto erfinv_transform, ErfinvTransform::Create());
  auto erfinv_res = (*erfinv_transform)(std::move(int_buf));
  EXPECT_FALSE(erfinv_res.ok());
  EXPECT_EQ(erfinv_res.status().code(), absl::StatusCode::kUnimplemented);
}

// Reduction transform tests adapted from
// /third_party/tensorflow/python/kernel_tests/math_ops/reduction_ops_test.py.
TEST(NumericalTransformsTest, ReduceAllAndAnyTransforms) {
  auto a = CreateBuffer<bool>({2, 2}, {true, false, true, true});

  ASSERT_OK_AND_ASSIGN(auto reduce_all,
                       ReduceAllTransform::Create(std::vector<int64_t>{0, 1}));
  ASSERT_OK_AND_ASSIGN(auto all_all, (*reduce_all)(a));
  EXPECT_EQ(all_all.shape(), Shape({}));
  EXPECT_EQ(GetData<bool>(all_all), std::vector<bool>({false}));

  ASSERT_OK_AND_ASSIGN(auto reduce_all_0,
                       ReduceAllTransform::Create(std::vector<int64_t>{0}));
  ASSERT_OK_AND_ASSIGN(auto all_0, (*reduce_all_0)(a));
  EXPECT_EQ(all_0.shape(), Shape({2}));
  EXPECT_EQ(GetData<bool>(all_0), std::vector<bool>({true, false}));

  ASSERT_OK_AND_ASSIGN(auto reduce_all_1,
                       ReduceAllTransform::Create(std::vector<int64_t>{1}));
  ASSERT_OK_AND_ASSIGN(auto all_1, (*reduce_all_1)(a));
  EXPECT_EQ(all_1.shape(), Shape({2}));
  EXPECT_EQ(GetData<bool>(all_1), std::vector<bool>({false, true}));

  ASSERT_OK_AND_ASSIGN(
      auto reduce_any,
      ReduceAnyTransform::Create(std::vector<int64_t>{1}, /*keep_dims=*/true));
  ASSERT_OK_AND_ASSIGN(auto any_1, (*reduce_any)(a));
  EXPECT_EQ(any_1.shape(), Shape({2, 1}));
  EXPECT_EQ(GetData<bool>(any_1), std::vector<bool>({true, true}));
}

TEST(NumericalTransformsTest, ReduceSumProdMinMaxMeanTransforms) {
  auto a = CreateBuffer<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  ASSERT_OK_AND_ASSIGN(auto sum_t,
                       ReduceSumTransform::Create(std::vector<int64_t>{1}));
  ASSERT_OK_AND_ASSIGN(auto sum_out, (*sum_t)(a));
  EXPECT_EQ(sum_out.shape(), Shape({2}));
  EXPECT_EQ(GetData<float>(sum_out), std::vector<float>({6.0f, 15.0f}));

  ASSERT_OK_AND_ASSIGN(auto prod_t,
                       ReduceProdTransform::Create(std::vector<int64_t>{1}));
  ASSERT_OK_AND_ASSIGN(auto prod_out, (*prod_t)(a));
  EXPECT_EQ(prod_out.shape(), Shape({2}));
  EXPECT_EQ(GetData<float>(prod_out), std::vector<float>({6.0f, 120.0f}));

  ASSERT_OK_AND_ASSIGN(auto min_t,
                       ReduceMinTransform::Create(std::vector<int64_t>{1}));
  ASSERT_OK_AND_ASSIGN(auto min_out, (*min_t)(a));
  EXPECT_EQ(min_out.shape(), Shape({2}));
  EXPECT_EQ(GetData<float>(min_out), std::vector<float>({1.0f, 4.0f}));

  ASSERT_OK_AND_ASSIGN(auto max_t,
                       ReduceMaxTransform::Create(std::vector<int64_t>{1}));
  ASSERT_OK_AND_ASSIGN(auto max_out, (*max_t)(a));
  EXPECT_EQ(max_out.shape(), Shape({2}));
  EXPECT_EQ(GetData<float>(max_out), std::vector<float>({3.0f, 6.0f}));

  ASSERT_OK_AND_ASSIGN(auto mean_t,
                       ReduceMeanTransform::Create(std::vector<int64_t>{1}));
  ASSERT_OK_AND_ASSIGN(auto mean_out, (*mean_t)(a));
  EXPECT_EQ(mean_out.shape(), Shape({2}));
  EXPECT_EQ(GetData<float>(mean_out), std::vector<float>({2.0f, 5.0f}));
}

TEST(NumericalTransformsTest, ReduceEuclideanNormTransform) {
  auto a = CreateBuffer<float>({2}, {3.0f, 4.0f});
  ASSERT_OK_AND_ASSIGN(auto euc_t, ReduceEuclideanNormTransform::Create(
                                       std::vector<int64_t>{0}));
  ASSERT_OK_AND_ASSIGN(auto euc_out, (*euc_t)(a));
  EXPECT_EQ(euc_out.shape(), Shape({}));
  EXPECT_NEAR(GetData<float>(euc_out)[0], 5.0f, 1e-5f);
}

TEST(NumericalTransformsTest, Reduce4DAlternatingAxesAndInvalidAxes) {
  std::vector<float> vals(16, 1.0f);
  auto a = CreateBuffer<float>({2, 2, 2, 2}, vals);

  ASSERT_OK_AND_ASSIGN(auto sum_t,
                       ReduceSumTransform::Create(std::vector<int64_t>{0, 2}));
  ASSERT_OK_AND_ASSIGN(auto out, (*sum_t)(a));
  EXPECT_EQ(out.shape(), Shape({2, 2}));
  EXPECT_EQ(GetData<float>(out), std::vector<float>({4.0f, 4.0f, 4.0f, 4.0f}));

  ASSERT_OK_AND_ASSIGN(auto dup_sum_t,
                       ReduceSumTransform::Create(std::vector<int64_t>{0, 0}));
  auto dup_status = (*dup_sum_t)(a).status();
  EXPECT_FALSE(dup_status.ok());
  EXPECT_THAT(dup_status.message(),
              testing::HasSubstr("Axes contains duplicate dimension"));
}

TEST(NumericalTransformsTest, ReduceSumTransformPropagateSignature) {
  ASSERT_OK_AND_ASSIGN(auto sum_t,
                       ReduceSumTransform::Create(std::vector<int64_t>{1}));
  const std::vector<ShapeInferenceValue> inputs = {
      HostBufferSpec{DType(DType::kF32), Shape({2, 3})},
  };
  ShapeInferenceContext context;
  context.inputs = inputs;

  ASSERT_OK(sum_t->PropagateSignature(context));
  ASSERT_EQ(context.outputs.size(), 1);
  EXPECT_EQ(context.outputs[0].spec().shape, Shape({2}));
}

TEST(NumericalTransformsTest,
     ReduceSumTransformPropagateSignatureUnknownDimension) {
  const std::vector<ShapeInferenceValue> inputs = {
      HostBufferSpec{DType(DType::kF32),
                     PartialShapeOrDie({Dim::Unknown(), Dim::Static(3)})},
  };

  ASSERT_OK_AND_ASSIGN(auto sum_axis1,
                       ReduceSumTransform::Create(std::vector<int64_t>{1}));
  ShapeInferenceContext ctx_axis1;
  ctx_axis1.inputs = inputs;
  ASSERT_OK(sum_axis1->PropagateSignature(ctx_axis1));
  ASSERT_EQ(ctx_axis1.outputs.size(), 1);
  EXPECT_EQ(ctx_axis1.outputs[0].spec().shape,
            PartialShapeOrDie({Dim::Unknown()}));

  ASSERT_OK_AND_ASSIGN(
      auto sum_keep_dims,
      ReduceSumTransform::Create(std::vector<int64_t>{1}, /*keep_dims=*/true));
  ShapeInferenceContext ctx_keep;
  ctx_keep.inputs = inputs;
  ASSERT_OK(sum_keep_dims->PropagateSignature(ctx_keep));
  ASSERT_EQ(ctx_keep.outputs.size(), 1);
  EXPECT_EQ(ctx_keep.outputs[0].spec().shape,
            PartialShapeOrDie({Dim::Unknown(), Dim::Static(1)}));

  ASSERT_OK_AND_ASSIGN(auto sum_axis0,
                       ReduceSumTransform::Create(std::vector<int64_t>{0}));
  ShapeInferenceContext ctx_axis0;
  ctx_axis0.inputs = inputs;
  ASSERT_OK(sum_axis0->PropagateSignature(ctx_axis0));
  ASSERT_EQ(ctx_axis0.outputs.size(), 1);
  EXPECT_EQ(ctx_axis0.outputs[0].spec().shape, Shape({3}));

  // Out-of-range axis still names the transform in the error.
  ASSERT_OK_AND_ASSIGN(auto sum_bad_axis,
                       ReduceSumTransform::Create(std::vector<int64_t>{2}));
  ShapeInferenceContext ctx_bad;
  ctx_bad.inputs = inputs;
  EXPECT_THAT(
      sum_bad_axis->PropagateSignature(ctx_bad),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("ReduceSumTransform: Invalid reduction "
                         "dimension (2 for input with 2 dimension(s))")));
}

TEST(NumericalTransformsTest, MaximumMinimumTransforms) {
  auto a = CreateBuffer<int32_t>({3}, {1, 10, -5});
  auto b = CreateBuffer<int32_t>({3}, {2, 5, -3});

  ASSERT_OK_AND_ASSIGN(auto max_t, MaximumTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto max_out, (*max_t)(a, b));
  EXPECT_EQ(GetData<int32_t>(max_out), std::vector<int32_t>({2, 10, -3}));

  ASSERT_OK_AND_ASSIGN(auto min_t, MinimumTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto min_out, (*min_t)(a, b));
  EXPECT_EQ(GetData<int32_t>(min_out), std::vector<int32_t>({1, 5, -5}));

  // Verify NaN propagation matching TensorFlow
  float nan = std::numeric_limits<float>::quiet_NaN();
  auto fa = CreateBuffer<float>({2}, {nan, 1.0f});
  auto fb = CreateBuffer<float>({2}, {1.0f, nan});
  ASSERT_OK_AND_ASSIGN(auto max_nan, (*max_t)(fa, fb));
  auto max_data = GetData<float>(max_nan);
  EXPECT_TRUE(std::isnan(max_data[0]));
  EXPECT_TRUE(std::isnan(max_data[1]));

  ASSERT_OK_AND_ASSIGN(auto min_nan, (*min_t)(fa, fb));
  auto min_data = GetData<float>(min_nan);
  EXPECT_TRUE(std::isnan(min_data[0]));
  EXPECT_TRUE(std::isnan(min_data[1]));
}

TEST(NumericalTransformsTest, MaximumTransformBroadcasting) {
  auto a = CreateBuffer<float>({2, 1}, {1.0f, 5.0f});
  auto b = CreateBuffer<float>({1, 3}, {2.0f, 0.0f, 6.0f});

  ASSERT_OK_AND_ASSIGN(auto transform, MaximumTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2, 3}));
  EXPECT_EQ(GetData<float>(out),
            std::vector<float>({2.0f, 1.0f, 6.0f, 5.0f, 5.0f, 6.0f}));
}

TEST(NumericalTransformsTest, MinimumTransformBroadcasting) {
  auto a = CreateBuffer<float>({2, 1}, {1.0f, 5.0f});
  auto b = CreateBuffer<float>({1, 3}, {2.0f, 0.0f, 6.0f});

  ASSERT_OK_AND_ASSIGN(auto transform, MinimumTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto out, (*transform)(std::move(a), std::move(b)));

  EXPECT_EQ(out.dtype(), DType(DType::kF32));
  EXPECT_EQ(out.shape(), Shape({2, 3}));
  EXPECT_EQ(GetData<float>(out),
            std::vector<float>({1.0f, 0.0f, 1.0f, 2.0f, 0.0f, 5.0f}));
}

TEST(NumericalTransformsTest, IgammaTransforms) {
  auto a = CreateBuffer<float>({2}, {1.0f, 2.0f});
  auto x = CreateBuffer<float>({2}, {1.0f, 3.0f});

  ASSERT_OK_AND_ASSIGN(auto igamma_t, IgammaTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto igamma_out, (*igamma_t)(a, x));
  auto igamma_data = GetData<float>(igamma_out);
  // igamma(1, 1) = 1 - exp(-1) ~= 0.63212055f
  EXPECT_NEAR(igamma_data[0], 0.63212055f, 1e-5f);

  ASSERT_OK_AND_ASSIGN(auto igammac_t, IgammacTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto igammac_out, (*igammac_t)(a, x));
  auto igammac_data = GetData<float>(igammac_out);
  // igammac(1, 1) = exp(-1) ~= 0.36787944f
  EXPECT_NEAR(igammac_data[0], 0.36787944f, 1e-5f);
  EXPECT_NEAR(igamma_data[0] + igammac_data[0], 1.0f, 1e-5f);
  EXPECT_NEAR(igamma_data[1] + igammac_data[1], 1.0f, 1e-5f);
}

TEST(NumericalTransformsTest, ZetaAndPolygammaTransforms) {
  auto x = CreateBuffer<float>({2}, {2.0f, 3.0f});
  auto q = CreateBuffer<float>({2}, {1.0f, 1.0f});

  ASSERT_OK_AND_ASSIGN(auto zeta_t, ZetaTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto zeta_out, (*zeta_t)(x, q));
  auto zeta_data = GetData<float>(zeta_out);
  // zeta(2, 1) = pi^2 / 6 ~= 1.644934f
  EXPECT_NEAR(zeta_data[0], 1.644934f, 1e-5f);

  auto n = CreateBuffer<float>({2}, {0.0f, 1.0f});
  auto px = CreateBuffer<float>({2}, {1.0f, 1.0f});
  ASSERT_OK_AND_ASSIGN(auto polygamma_t, PolygammaTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto polygamma_out, (*polygamma_t)(n, px));
  auto polygamma_data = GetData<float>(polygamma_out);
  // polygamma(0, 1) = digamma(1) = -euler_gamma ~= -0.57721566f
  // polygamma(1, 1) = zeta(2, 1) = pi^2 / 6 ~= 1.644934f
  EXPECT_NEAR(polygamma_data[0], -0.57721566f, 1e-5f);
  EXPECT_NEAR(polygamma_data[1], 1.644934f, 1e-5f);
}

TEST(NumericalTransformsTest, XdivyXlog1pyXlogyTransforms) {
  auto x = CreateBuffer<float>({3}, {0.0f, 2.0f, 0.0f});
  auto y_div = CreateBuffer<float>({3}, {0.0f, 4.0f, 5.0f});
  ASSERT_OK_AND_ASSIGN(auto xdivy_t, XdivyTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto xdivy_out, (*xdivy_t)(x, y_div));
  EXPECT_EQ(GetData<float>(xdivy_out), std::vector<float>({0.0f, 0.5f, 0.0f}));

  auto y_log1p = CreateBuffer<float>({3}, {-1.0f, 0.0f, 2.0f});
  ASSERT_OK_AND_ASSIGN(auto xlog1py_t, Xlog1pyTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto xlog1py_out, (*xlog1py_t)(x, y_log1p));
  auto xlog1py_data = GetData<float>(xlog1py_out);
  EXPECT_EQ(xlog1py_data[0], 0.0f);
  EXPECT_EQ(xlog1py_data[1], 0.0f);
  EXPECT_EQ(xlog1py_data[2], 0.0f);

  auto y_log = CreateBuffer<float>({3}, {0.0f, 1.0f, 2.0f});
  ASSERT_OK_AND_ASSIGN(auto xlogy_t, XlogyTransform::Create());
  ASSERT_OK_AND_ASSIGN(auto xlogy_out, (*xlogy_t)(x, y_log));
  auto xlogy_data = GetData<float>(xlogy_out);
  EXPECT_EQ(xlogy_data[0], 0.0f);
  EXPECT_EQ(xlogy_data[1], 0.0f);
  EXPECT_EQ(xlogy_data[2], 0.0f);

  // Verify broadcasting with zero x
  auto x_bcast = CreateBuffer<float>({2, 1}, {0.0f, 0.0f});
  auto y_bcast = CreateBuffer<float>({1, 3}, {0.0f, 1.0f, 2.0f});
  ASSERT_OK_AND_ASSIGN(auto xlogy_bcast, (*xlogy_t)(x_bcast, y_bcast));
  EXPECT_EQ(xlogy_bcast.shape(), Shape({2, 3}));
  EXPECT_EQ(GetData<float>(xlogy_bcast),
            std::vector<float>({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
}

}  // namespace
}  // namespace xla_serving::data::kernel
