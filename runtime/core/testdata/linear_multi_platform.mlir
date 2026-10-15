// Exported from "linear_model.py" after adding 'native_serialization_platform': ['cpu', 'tpu'],
module @linear_model
    attributes {jax.uses_shape_polymorphism = true, mhlo.num_partitions = 4 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(
    %arg0: tensor<i32> {jax.global_constant = "_platform_index"},
    %arg1: tensor<2xf32> {mhlo.sharding = "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"},
    %arg2: tensor<2x2xf32> {mhlo.sharding = "{devices=[2,2]<=[4]}"},
    %arg3: tensor<4x2xf32> {mhlo.sharding = "{devices=[2,1,2]<=[4] last_tile_dim_replicate}"})
    -> (tensor<4x2xf32> {jax.result_info = "result"}) {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %0 = "stablehlo.case"(%arg0) ({
      stablehlo.return %c_0 : tensor<i32>
    }, {
      stablehlo.return %c : tensor<i32>
    }) : (tensor<i32>) -> tensor<i32>
    %1 = "stablehlo.case"(%0) ({
      %5 = stablehlo.dot_general %arg3, %arg2, contracting_dims = [1] x [0] : (tensor<4x2xf32>, tensor<2x2xf32>) -> tensor<4x2xf32>
      stablehlo.return %5 : tensor<4x2xf32>
    }, {
      %5 = stablehlo.dot_general %arg3, %arg2, contracting_dims = [1] x [0] : (tensor<4x2xf32>, tensor<2x2xf32>) -> tensor<4x2xf32>
      stablehlo.return %5 : tensor<4x2xf32>
    }) : (tensor<i32>) -> tensor<4x2xf32>
    %2 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1x2xf32>) -> tensor<4x2xf32>
    %4 = stablehlo.add %1, %3 : tensor<4x2xf32>
    return %4 : tensor<4x2xf32>
  }
}