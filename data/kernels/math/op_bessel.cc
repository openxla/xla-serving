/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

#include "data/kernels/numerical_transforms.h"  // from @xla_serving
#include "data/kernels/op_common.h"  // from @xla_serving
#include "xla/python/ifrt/dtype.h"  // from @xla

// This file implements the Bessel Transform operators.
// It is branched from
// /third_party/tensorflow/core/kernels/special_math/special_math_op_bessel.cc.
namespace xla_serving::data::kernel {

ND_REGISTER_UNARY_OP(BesselI0Transform,
                     {xla::ifrt::DType::kF16, xla::ifrt::DType::kF32,
                      xla::ifrt::DType::kF64});

ND_REGISTER_UNARY_OP(BesselI1Transform,
                     {xla::ifrt::DType::kF16, xla::ifrt::DType::kF32,
                      xla::ifrt::DType::kF64});

ND_REGISTER_UNARY_OP(BesselI0eTransform,
                     {xla::ifrt::DType::kF16, xla::ifrt::DType::kF32,
                      xla::ifrt::DType::kF64});

ND_REGISTER_UNARY_OP(BesselI1eTransform,
                     {xla::ifrt::DType::kF16, xla::ifrt::DType::kF32,
                      xla::ifrt::DType::kF64});

ND_REGISTER_UNARY_OP(BesselK0Transform,
                     {xla::ifrt::DType::kF16, xla::ifrt::DType::kF32,
                      xla::ifrt::DType::kF64});

ND_REGISTER_UNARY_OP(BesselK1Transform,
                     {xla::ifrt::DType::kF16, xla::ifrt::DType::kF32,
                      xla::ifrt::DType::kF64});

ND_REGISTER_UNARY_OP(BesselK0eTransform,
                     {xla::ifrt::DType::kF16, xla::ifrt::DType::kF32,
                      xla::ifrt::DType::kF64});

ND_REGISTER_UNARY_OP(BesselK1eTransform,
                     {xla::ifrt::DType::kF16, xla::ifrt::DType::kF32,
                      xla::ifrt::DType::kF64});

ND_REGISTER_UNARY_OP(BesselJ0Transform,
                     {xla::ifrt::DType::kF16, xla::ifrt::DType::kF32,
                      xla::ifrt::DType::kF64});

ND_REGISTER_UNARY_OP(BesselJ1Transform,
                     {xla::ifrt::DType::kF16, xla::ifrt::DType::kF32,
                      xla::ifrt::DType::kF64});

ND_REGISTER_UNARY_OP(BesselY0Transform,
                     {xla::ifrt::DType::kF16, xla::ifrt::DType::kF32,
                      xla::ifrt::DType::kF64});

ND_REGISTER_UNARY_OP(BesselY1Transform,
                     {xla::ifrt::DType::kF16, xla::ifrt::DType::kF32,
                      xla::ifrt::DType::kF64});

}  // namespace xla_serving::data::kernel
