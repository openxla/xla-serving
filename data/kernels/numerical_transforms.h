#ifndef XLA_SERVING_DATA_KERNELS_NUMERICAL_TRANSFORMS_H_
#define XLA_SERVING_DATA_KERNELS_NUMERICAL_TRANSFORMS_H_

#include <memory>

#include "data/kernels/bitwise/op_functors.h"  // from @xla_serving
#include "data/kernels/math/op_functors.h"  // from @xla_serving
#include "data/kernels/math/reduction_ops_common.h"  // from @xla_serving
#include "data/kernels/math/scan_ops.h"  // from @xla_serving
#include "data/kernels/op_common.h"  // from @xla_serving
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "data/processor/cc/api/transform_context.h"  // from @xla_serving
#include "data/processor/cc/core/transform.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving

namespace xla_serving::data::kernel {

// Binary Transforms.
ND_DECLARE_BINARY_OP(PowTransform, ::xla_serving::data::kernel::functor::pow);
ND_DECLARE_BINARY_OP(IgammaTransform, ::xla_serving::data::kernel::functor::igamma);
ND_DECLARE_BINARY_OP(IgammacTransform, ::xla_serving::data::kernel::functor::igammac);
ND_DECLARE_BINARY_OP(MaximumTransform, ::xla_serving::data::kernel::functor::maximum);
ND_DECLARE_BINARY_OP(MinimumTransform, ::xla_serving::data::kernel::functor::minimum);
ND_DECLARE_BINARY_OP(PolygammaTransform,
                     ::xla_serving::data::kernel::functor::polygamma);
ND_DECLARE_BINARY_OP(XdivyTransform, ::xla_serving::data::kernel::functor::xdivy);
ND_DECLARE_BINARY_OP(Xlog1pyTransform, ::xla_serving::data::kernel::functor::xlog1py);
ND_DECLARE_BINARY_OP(XlogyTransform, ::xla_serving::data::kernel::functor::xlogy);
ND_DECLARE_BINARY_OP(ZetaTransform, ::xla_serving::data::kernel::functor::zeta);
ND_DECLARE_BINARY_OP(LogicalAndTransform,
                     ::xla_serving::data::kernel::functor::logical_and);
ND_DECLARE_BINARY_OP(LogicalOrTransform,
                     ::xla_serving::data::kernel::functor::logical_or);
ND_DECLARE_BINARY_OP(LogicalXorTransform,
                     ::xla_serving::data::kernel::functor::logical_xor);
ND_DECLARE_BINARY_OP(AddTransform, ::xla_serving::data::kernel::functor::add);
ND_DECLARE_BINARY_OP(SubtractTransform, ::xla_serving::data::kernel::functor::sub);
ND_DECLARE_BINARY_OP(MultiplyTransform, ::xla_serving::data::kernel::functor::mul);
ND_DECLARE_BINARY_OP(MultiplyNoNanTransform,
                     ::xla_serving::data::kernel::functor::mul_no_nan);
ND_DECLARE_BINARY_OP(DivideTransform, ::xla_serving::data::kernel::functor::div);
ND_DECLARE_BINARY_OP(DivideNoNanTransform,
                     ::xla_serving::data::kernel::functor::div_no_nan);
ND_DECLARE_BINARY_OP(TrueDivTransform,
                     ::xla_serving::data::kernel::functor::true_div);
ND_DECLARE_BINARY_OP(FloorDivTransform,
                     ::xla_serving::data::kernel::functor::floor_div);
ND_DECLARE_BINARY_OP(FloorModTransform,
                     ::xla_serving::data::kernel::functor::floor_mod);
ND_DECLARE_BINARY_OP(Atan2Transform, ::xla_serving::data::kernel::functor::atan2);

// Unary Transforms.
ND_DECLARE_UNARY_OP(LogicalNotTransform,
                    ::xla_serving::data::kernel::functor::logical_not);
ND_DECLARE_UNARY_OP(SinTransform, ::xla_serving::data::kernel::functor::sin);
ND_DECLARE_UNARY_OP(CosTransform, ::xla_serving::data::kernel::functor::cos);
ND_DECLARE_UNARY_OP(TanTransform, ::xla_serving::data::kernel::functor::tan);
ND_DECLARE_UNARY_OP(AsinTransform, ::xla_serving::data::kernel::functor::asin);
ND_DECLARE_UNARY_OP(AcosTransform, ::xla_serving::data::kernel::functor::acos);
ND_DECLARE_UNARY_OP(AtanTransform, ::xla_serving::data::kernel::functor::atan);
ND_DECLARE_UNARY_OP(SinhTransform, ::xla_serving::data::kernel::functor::sinh);
ND_DECLARE_UNARY_OP(CoshTransform, ::xla_serving::data::kernel::functor::cosh);
ND_DECLARE_UNARY_OP(TanhTransform, ::xla_serving::data::kernel::functor::tanh);
ND_DECLARE_UNARY_OP(AsinhTransform, ::xla_serving::data::kernel::functor::asinh);
ND_DECLARE_UNARY_OP(AcoshTransform, ::xla_serving::data::kernel::functor::acosh);
ND_DECLARE_UNARY_OP(AtanhTransform, ::xla_serving::data::kernel::functor::atanh);
ND_DECLARE_UNARY_OP(SqrtTransform, ::xla_serving::data::kernel::functor::sqrt);
ND_DECLARE_UNARY_OP(AbsTransform, ::xla_serving::data::kernel::functor::abs);
ND_DECLARE_UNARY_OP(CeilTransform, ::xla_serving::data::kernel::functor::ceil);
ND_DECLARE_UNARY_OP(ExpTransform, ::xla_serving::data::kernel::functor::exp);
ND_DECLARE_UNARY_OP(Expm1Transform, ::xla_serving::data::kernel::functor::expm1);
ND_DECLARE_UNARY_OP(FloorTransform, ::xla_serving::data::kernel::functor::floor);
ND_DECLARE_UNARY_OP(ReciprocalTransform,
                    ::xla_serving::data::kernel::functor::reciprocal);
ND_DECLARE_UNARY_OP(LogTransform, ::xla_serving::data::kernel::functor::log);
ND_DECLARE_UNARY_OP(Log1pTransform, ::xla_serving::data::kernel::functor::log1p);
ND_DECLARE_UNARY_OP(NegativeTransform, ::xla_serving::data::kernel::functor::negative);
ND_DECLARE_UNARY_OP(RintTransform, ::xla_serving::data::kernel::functor::rint);
ND_DECLARE_UNARY_OP(RoundTransform, ::xla_serving::data::kernel::functor::round);
ND_DECLARE_UNARY_OP(RsqrtTransform, ::xla_serving::data::kernel::functor::rsqrt);
ND_DECLARE_UNARY_OP(SignTransform, ::xla_serving::data::kernel::functor::sign);
ND_DECLARE_UNARY_OP(SigmoidTransform, ::xla_serving::data::kernel::functor::sigmoid);
ND_DECLARE_UNARY_OP(SquareTransform, ::xla_serving::data::kernel::functor::square);
ND_DECLARE_UNARY_OP(ConjTransform, ::xla_serving::data::kernel::functor::conj);
ND_DECLARE_UNARY_OP(ComplexAbsTransform, ::xla_serving::data::kernel::functor::abs);

// Logical & Bitwise
ND_DECLARE_BINARY_OP(BitwiseAndTransform,
                     ::xla_serving::data::kernel::functor::bitwise_and);
ND_DECLARE_BINARY_OP(BitwiseOrTransform,
                     ::xla_serving::data::kernel::functor::bitwise_or);
ND_DECLARE_BINARY_OP(BitwiseXorTransform,
                     ::xla_serving::data::kernel::functor::bitwise_xor);
ND_DECLARE_BINARY_OP(LeftShiftTransform,
                     ::xla_serving::data::kernel::functor::left_shift);
ND_DECLARE_BINARY_OP(RightShiftTransform,
                     ::xla_serving::data::kernel::functor::right_shift);
ND_DECLARE_UNARY_OP(InvertTransform, ::xla_serving::data::kernel::functor::invert);
ND_DECLARE_UNARY_OP(IsFiniteTransform, ::xla_serving::data::kernel::functor::isfinite);
ND_DECLARE_UNARY_OP(IsInfTransform, ::xla_serving::data::kernel::functor::isinf);
ND_DECLARE_UNARY_OP(IsNanTransform, ::xla_serving::data::kernel::functor::isnan);

// Special Math / Advanced Functions
ND_DECLARE_UNARY_OP(DawsnTransform, ::xla_serving::data::kernel::functor::dawsn);
ND_DECLARE_UNARY_OP(DigammaTransform, ::xla_serving::data::kernel::functor::digamma);
ND_DECLARE_UNARY_OP(ErfTransform, ::xla_serving::data::kernel::functor::erf);
ND_DECLARE_UNARY_OP(ErfcTransform, ::xla_serving::data::kernel::functor::erfc);
ND_DECLARE_UNARY_OP(ErfinvTransform, ::xla_serving::data::kernel::functor::erfinv);
ND_DECLARE_UNARY_OP(ExpintTransform, ::xla_serving::data::kernel::functor::expint);
ND_DECLARE_UNARY_OP(FresnelCosTransform,
                    ::xla_serving::data::kernel::functor::fresnel_cos);
ND_DECLARE_UNARY_OP(FresnelSinTransform,
                    ::xla_serving::data::kernel::functor::fresnel_sin);
ND_DECLARE_UNARY_OP(LgammaTransform, ::xla_serving::data::kernel::functor::lgamma);
ND_DECLARE_UNARY_OP(NdtriTransform, ::xla_serving::data::kernel::functor::ndtri);
ND_DECLARE_UNARY_OP(SpenceTransform, ::xla_serving::data::kernel::functor::spence);

// Bessel Functions
ND_DECLARE_UNARY_OP(BesselI0Transform, ::xla_serving::data::kernel::functor::bessel_i0);
ND_DECLARE_UNARY_OP(BesselI0eTransform,
                    ::xla_serving::data::kernel::functor::bessel_i0e);
ND_DECLARE_UNARY_OP(BesselI1Transform, ::xla_serving::data::kernel::functor::bessel_i1);
ND_DECLARE_UNARY_OP(BesselI1eTransform,
                    ::xla_serving::data::kernel::functor::bessel_i1e);
ND_DECLARE_UNARY_OP(BesselK0Transform, ::xla_serving::data::kernel::functor::bessel_k0);
ND_DECLARE_UNARY_OP(BesselK0eTransform,
                    ::xla_serving::data::kernel::functor::bessel_k0e);
ND_DECLARE_UNARY_OP(BesselK1Transform, ::xla_serving::data::kernel::functor::bessel_k1);
ND_DECLARE_UNARY_OP(BesselK1eTransform,
                    ::xla_serving::data::kernel::functor::bessel_k1e);
ND_DECLARE_UNARY_OP(BesselJ0Transform, ::xla_serving::data::kernel::functor::bessel_j0);
ND_DECLARE_UNARY_OP(BesselJ1Transform, ::xla_serving::data::kernel::functor::bessel_j1);
ND_DECLARE_UNARY_OP(BesselY0Transform, ::xla_serving::data::kernel::functor::bessel_y0);
ND_DECLARE_UNARY_OP(BesselY1Transform, ::xla_serving::data::kernel::functor::bessel_y1);

class ND_EXPORT_SYMBOL MatMulTransform : public xla_serving::data::composite::TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<MatMulTransform>> Create();
  absl::StatusOr<xla_serving::HostBuffer> operator()(xla_serving::HostBuffer a,
                                             xla_serving::HostBuffer b);
  absl::StatusOr<xla_serving::data::composite::TransformConfig> ToProto()
      const override;

 protected:
  absl::Status PropagateSignatureImpl(
      xla_serving::data::composite::api::ShapeInferenceContext& context)
      const override;
};

}  // namespace xla_serving::data::kernel

#endif  // XLA_SERVING_DATA_KERNELS_NUMERICAL_TRANSFORMS_H_
