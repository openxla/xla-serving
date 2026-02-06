module @jit_jax_model_fn attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x2xf32> {mhlo.layout_mode = "default"}, %arg1: tensor<2x8xf32> {mhlo.layout_mode = "default"}) -> (tensor<2x8xf32> {jax.result_info = "[0]", mhlo.layout_mode =
 "default"}, tensor<2x8xf32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<2x2xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2x8xf32>
    %2 = stablehlo.subtract %0, %1 : tensor<2x8xf32>
    return %0, %2 : tensor<2x8xf32>, tensor<2x8xf32>
  }
}
