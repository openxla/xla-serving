// This is a shardy model such that shardings are propagated to parameters and outputs.
module @jit_jax_model_fn attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  func.func @main(%platform_index: tensor<i32>, %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}, {"model"}]>}, %arg1: tensor<8x8xf32>, %arg2: tensor<8x16xf32>) -> tensor<8x16xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
    %1 = stablehlo.dot_general %0, %arg2, contracting_dims = [1] x [0] :
      (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
    return %1 : tensor<8x16xf32>
  }
  sdy.mesh @mesh = <["data"=2, "model"=2]>
}
