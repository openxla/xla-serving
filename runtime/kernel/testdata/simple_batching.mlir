module @jit_jax_model_fn attributes {jax.uses_shape_polymorphism = true, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x8xf32> {mhlo.layout_mode = "default"}, %arg1: tensor<?x2xf32> {mhlo.layout_mode = "default"}) -> (tensor<?x8xf32> {jax.result_info = "[0]", mhlo.layout_mode =
 "default"}) {
    %0 = stablehlo.dot_general %arg1, %arg0, contracting_dims = [1] x [0] : (tensor<?x2xf32>, tensor<2x8xf32>) -> tensor<?x8xf32>
    return %0 : tensor<?x8xf32>
  }
}
