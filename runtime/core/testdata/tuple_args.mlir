 module @jit_fn {
    func.func public @main(%arg0: tuple<tensor<4xf32>, tensor<4xf32>>) -> (tuple<tensor<4xf32>, tensor<4xf32>>) {
      return %arg0: tuple<tensor<4xf32>, tensor<4xf32>>
    }
  }