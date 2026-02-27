// RUN: stablehlo-opt -tpu-input-r1-optimization-pass='variable-arg-indices-list=1' %s | FileCheck %s

// CHECK-LABEL: module @jit_f attributes {jax.uses_shape_polymorphism = true} {
// CHECK:          func.func public @main(%arg0: tensor<6xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<f32>) -> (tensor<3x3xf32> {jax.result_info = ""}) {
// CHECK:            %[[OUT_0:.+]] = stablehlo.reshape %arg0 : (tensor<6xf32>) -> tensor<3x2xf32>
// CHECK:            %[[OUT_c:.+]] = stablehlo.constant dense<3> : tensor<i64>
// CHECK:            %[[OUT_c_0:.+]] = stablehlo.constant dense<2> : tensor<i64>
// CHECK:            %[[OUT_c_1:.+]] = stablehlo.constant dense<true> : tensor<i1>
// CHECK:            %[[OUT_1:.+]] = stablehlo.dot_general %[[OUT_0]], %arg1, contracting_dims = [1] x [0] : (tensor<3x2xf32>, tensor<2x3xf32>) -> tensor<3x3xf32>
// CHECK:            %[[OUT_2:.+]] = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<3x3xf32>
// CHECK:            %[[OUT_3:.+]] = stablehlo.add %[[OUT_1]], %[[OUT_2]] : tensor<3x3xf32>
// CHECK:            return %[[OUT_3:.+]] : tensor<3x3xf32>
// CHECK:          }
// CHECK:        }
module @jit_f attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg0: tensor<3x2xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<f32>) -> (tensor<3x3xf32> {jax.result_info = ""}) {
    %c = stablehlo.constant dense<3> : tensor<i64>
    %c_0 = stablehlo.constant dense<2> : tensor<i64>
    %c_1 = stablehlo.constant dense<true> : tensor<i1>
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<3x2xf32>, tensor<2x3xf32>) -> tensor<3x3xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<3x3xf32>
    %2 = stablehlo.add %0, %1 : tensor<3x3xf32>
    return %2 : tensor<3x3xf32>
  }
}