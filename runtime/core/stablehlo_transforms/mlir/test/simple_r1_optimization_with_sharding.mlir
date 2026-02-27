// RUN: stablehlo-opt -tpu-input-r1-optimization-pass='variable-arg-indices-list=1 enable_serialize_sharding=false' %s | FileCheck %s

// CHECK-LABEL: module @jit__linear_model attributes {jax.uses_shape_polymorphism = true, mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
// CHECK:          func.func public @main(%arg0: tensor<2xf32> {mhlo.sharding = "{devices=[2,1]1,0}"}, %arg1: tensor<2x2xf32> {mhlo.sharding = "{devices=[2,1]0,1}"}, %arg2: tensor<4xf32> {mhlo.sharding = "{devices=[2]1,0}"}) -> (tensor<2x2xf32> {jax.result_info = "result['y']"})
// CHECK:             %[[OUT_0:.+]] = stablehlo.reshape %arg2 : (tensor<4xf32>) -> tensor<2x2xf32>
// CHECK:             %[[OUT_c:.+]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK:             %[[OUT_c0:.+]] = stablehlo.constant dense<2> : tensor<i32>
// CHECK:             %[[OUT_c1:.+]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK:             %[[OUT_c2:.+]] = stablehlo.constant dense<true> : tensor<i1>
// CHECK:             %[[OUT_1:.+]] = stablehlo.dot_general %[[OUT_0]], %arg1, contracting_dims = [1] x [0] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK:             %[[OUT_2:.+]] = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
// CHECK:             %[[OUT_3:.+]] = stablehlo.broadcast_in_dim %[[OUT_2]], dims = [0, 1] : (tensor<1x2xf32>) -> tensor<2x2xf32>
// CHECK:             %[[OUT_4:.+]] = stablehlo.add %[[OUT_1]], %[[OUT_3]] : tensor<2x2xf32>
// CHECK:             return %[[OUT_4]] : tensor<2x2xf32>

module @jit__linear_model attributes {jax.uses_shape_polymorphism = true, mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2xf32> {mhlo.sharding = "{devices=[2,1]1,0}"}, %arg1: tensor<2x2xf32> {mhlo.sharding = "{devices=[2,1]0,1}"}, %arg2: tensor<2x2xf32> {mhlo.sharding = "{devices=[2,1,1]1,0}"}) -> (tensor<2x2xf32> {jax.result_info = "result['y']"}) {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<2> : tensor<i32>
    %c_1 = stablehlo.constant dense<0> : tensor<i32>
    %c_2 = stablehlo.constant dense<true> : tensor<i1>
    %0 = stablehlo.dot_general %arg2, %arg1, contracting_dims = [1] x [0] : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x2xf32>) -> tensor<2x2xf32>
    %3 = stablehlo.add %0, %2 : tensor<2x2xf32>
    return %3 : tensor<2x2xf32>
  }
}