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

// This file implements the FresnelCosTransform operator.
// It is branched from
// /third_party/tensorflow/core/kernels/special_math/special_math_op_fresnel_cos.cc.
namespace xla_serving::data::kernel {

ND_REGISTER_UNARY_OP(FresnelCosTransform,
                     {xla::ifrt::DType::kF32, xla::ifrt::DType::kF64});

}  // namespace xla_serving::data::kernel
