load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = [
    "//learning/infra/mira/experimental/orbax_model:__subpackages__",
    "@neptune//runtime:__subpackages__",
])

cc_library(
    name = "stablehlo_tpu_optimizer",
    srcs = ["stablehlo_tpu_optimizer.cc"],
    hdrs = ["stablehlo_tpu_optimizer.h"],
    deps = [
        ":tpu_input_r1_optimization",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@neptune//runtime/core:stablehlo_optimizer",
        "@xla//xla/mlir/utils:error_util",
    ],
)

cc_library(
    name = "tpu_input_r1_optimization",
    srcs = ["tpu_input_r1_optimization.cc"],
    hdrs = ["tpu_input_r1_optimization.h"],
    deps = [
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@neptune//runtime/libs/mlir:dump_mlir_util",
        "@neptune//runtime/libs/mlir:utils",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@stablehlo//:stablehlo_ops",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/ir:hlo_sharding",
        "@xla//xla/hlo/parser:hlo_parser",
        "@xla//xla/pjrt/proto:compile_options_proto_cc",
        "@xla//xla/service/spmd/shardy:utils",
        "@xla//xla/service/spmd/shardy/stablehlo_round_trip:export_shardings",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
    alwayslink = 1,
)
