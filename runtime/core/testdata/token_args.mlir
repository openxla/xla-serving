module @jit_fn {
    func.func public @main(%arg0: !stablehlo.token, %arg1: tensor<f32>) -> (!stablehlo.token, tensor<f32>) {
      return %arg0, %arg1: !stablehlo.token, tensor<f32>
    }
  }