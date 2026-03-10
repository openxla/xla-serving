load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

package(default_visibility = ["@neptune//runtime/core/stablehlo_transforms:__subpackages__"])

exports_files(["run_lit.sh"])

cc_binary(
    name = "stablehlo-opt",
    srcs = [
        "stablehlo-opt.cc",
    ],
    deps = [
        # Register TpuInputR1OptimizationPass.
        "//learning/infra/mira/experimental/orbax_model/tensorflow/stablehlo_transforms/tpu:tpu_input_r1_optimization",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:RegisterAllDialects",  # buildcleaner: keep
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:ShapeDialect",
        "@stablehlo//:register",
    ],
)
