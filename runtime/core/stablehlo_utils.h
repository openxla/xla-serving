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

#ifndef XLA_SERVING_RUNTIME_STABLEHLO_UTILS_H_
#define XLA_SERVING_RUNTIME_STABLEHLO_UTILS_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "runtime/libs/mlir/utils.h"  // from @xla_serving
#include "xla/hlo/ir/hlo_sharding.h"  // from @xla
#include "xla/layout.h"  // from @xla
#include "xla/pjrt/proto/compile_options.pb.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla

namespace xla_serving {

enum class Placement { kHost = 0, kDevice = 1 };

// Represents the shape of an input or output. Any dimension that is not set is
// assumed to be dynamic. We just use the same type that we get out of the MLIR
// library for this, it can be assumed to behave like a
// std::vector<std::optional<int64_t>>, where std::nullopt means dynamic.
using ArgShape = sharding_internal::MlirShape;

// Converts an `ArgShape` to `xla::ifrt::Shape` for the purposes of checkpoint
// loading. The result will be fake if the shape is dynamic, so don't use this
// function for anything else.
xla::ifrt::Shape ToIfrtShape(const ArgShape& shape);

struct ArgSpec {
  xla::ifrt::DType dtype;
  ArgShape shape;
  std::optional<xla::ifrt::Shape> ifrt_shape;
  xla::HloSharding sharding;
  std::optional<xla::Layout> layout;
  std::optional<Placement> placement = std::nullopt;

  xla::ifrt::Shape GetIfrtShape() const {
    if (ifrt_shape.has_value()) {
      return *ifrt_shape;
    }
    return ToIfrtShape(shape);
  }

  bool operator==(const ArgSpec& other) const {
    return dtype == other.dtype && shape == other.shape &&
           sharding == other.sharding;
  }

  // Before c++ 20, the compiler is not smart enough to deduce the
  // operator!= from operator==, so we need to implement it since OSS uses
  // c++ 17.
  // TODO: Remove this once we move to c++ 20.
  bool operator!=(const ArgSpec& other) const { return !(*this == other); }

  // Checks if the `dtype` and `shape` of two ArgSpec are equal.
  bool IsDTypeAndShapeEqual(const ArgSpec& other) const {
    return dtype == other.dtype && shape == other.shape;
  }

  std::string DebugString() const {
    std::string shape_str = "{";
    for (int i = 0; i < shape.size(); ++i) {
      if (i > 0 && i < shape.size()) {
        absl::StrAppend(&shape_str, ", ");
      }
      if (shape[i].has_value()) {
        absl::StrAppend(&shape_str, *shape[i]);
      } else {
        absl::StrAppend(&shape_str, "nullopt");
      }
    }
    absl::StrAppend(&shape_str, "}");
    return absl::StrCat("dtype: ", dtype, ", shape: ", shape_str,
                        ", sharding: ", sharding.ToString());
  }
};

struct StableHloIOSpec {
  std::vector<ArgSpec> inputs;
  std::vector<ArgSpec> outputs;
};

absl::StatusOr<bool> UsesPolymorphicShape(mlir::ModuleOp module);

// Gets StableHloIOSpec from the MLIR module.
absl::StatusOr<StableHloIOSpec> GetStableHloIOSpec(
    mlir::ModuleOp module,
    xla::CompileOptionsProto* absl_nullable compile_options);

// Sets the platform index argument if the module is compiled for multiple
// platforms, and then erases the argument. The platform index argument will be
// set to the index of the `platform_name` in the `supported_platforms` list.
// This function is almost a clone of the implementation in
// /third_party/tensorflow/compiler/tf2xla/kernels/xla_call_module_loader.cc.
absl::Status SetPlatformIndex(mlir::ModuleOp module,
                              absl::Span<const std::string> supported_platforms,
                              absl::string_view platform_name);

// Refines dynamic shapes of an MLIR module if needed.
//
// This function refines the dynamic shapes of the "main" function's inputs
// (weights and user inputs) when they have unknown dimensions for user inputs.
// The refinement is based on the static shapes of weights and inputs, and it
// assumes that the rank of the dynamic shape is known. Dynamic shapes with
// unknown ranks are not supported.
//
// Args:
//   - `module`: The MLIR module to refine.
//   - `weights`: The dynamic weight specifications (`ArgSpec`) of the
//      model. The spec is nullopt if the weight is unused by StableHLO(s).
//   - `input_shapes`: The static shapes of the inputs (excluding weights),
//      regardless of whether they are used by the "main" function.
//   - `input_dtypes`: The data types of the inputs (excluding weights),
//      regardless of whether they are used by the "main" function.
//   - `module_kept_var_idx`: Specifies which of the weights and user inputs
//      need to be passed to the "main" function as arguments.
absl::Status RefineShapesIfNeeded(
    mlir::ModuleOp module, absl::Span<const std::optional<ArgSpec>> weights,
    absl::Span<const xla::ifrt::Shape> input_shapes,
    absl::Span<const xla::ifrt::DType> input_dtypes,
    absl::Span<const int> module_kept_var_idx);

// Validates an IFRT DType is supported.
absl::Status ValidateIOType(xla::ifrt::DType dtype);

// Validates the IO specs between StableHLO and JAX model are the same, and
// the DTypes of input and output are supported.
// Args:
//   - `stablehlo_arg_specs`: The IO specs of the main function from StableHLO
//      MLIR module. This a subset of JAX model function inputs, since some
//      of JAX function inputs may not be used by the StableHLO.
//   - `model_arg_specs`: The IO specs of the JAX model function regardless
//      of whether they are used by the StableHLO.
//   - `module_kept_var_idx`: Specifies which of the JAX model function input
//      args are required to be passed to the StableHLO function. When empty,
//      it is assumed that all JAX model function input args are required to be
//      passed to the StableHLO function.
absl::Status ValidateIOSpecs(absl::Span<const ArgSpec> stablehlo_arg_specs,
                             absl::Span<const ArgSpec> model_arg_specs,
                             absl::Span<const int> module_kept_var_idx);

// Verifies the StableHLO input shape matches the actual input shape. This
// function assumes input shapes are fully defined.
absl::Status InputShapeEqual(ArgShape stablehlo_input_shape,
                             xla::ifrt::Shape input_shape);

// Converts an `ArgShape` to `xla::ifrt::Shape` for the purposes of checkpoint
// loading. The result will be fake if the shape is dynamic, so don't use this
// function for anything else.
xla::ifrt::Shape ToIfrtShape(const ArgShape& shape);

};  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_STABLEHLO_UTILS_H_
