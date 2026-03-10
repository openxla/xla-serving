load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = [
    "//learning/infra/mira/experimental/orbax_model:__subpackages__",
    "@neptune//:__subpackages__",
])

cc_library(
    name = "stablehlo_loader",
    srcs = ["stablehlo_loader.cc"],
    hdrs = ["stablehlo_loader.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:RegisterAllDialects",
        "@llvm-project//mlir:RegisterAllPasses",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:TensorDialect",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:register",
        "@stablehlo//:stablehlo_ops",
        "@xla//xla/pjrt:mlir_to_hlo",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "executable_build_utils",
    srcs = ["executable_build_utils.cc"],
    hdrs = ["executable_build_utils.h"],
    defines = ["JSV_ON_GCE"],
    deps = [
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@llvm-project//mlir:IR",
        "@neptune//runtime/libs/ifrt/persistent_compilation_cache:ifrt_persistent_compilation_cache",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt/proto:compile_options_proto_cc",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/ifrt/hlo:hlo_program",
        "@xla//xla/python/pjrt_ifrt:xla_ifrt",
        "@xla//xla/service:compilation_environments",
        "@xla//xla/service:computation_placer_hdr",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "executable_build_utils_gce",
    srcs = ["executable_build_utils.cc"],
    hdrs = ["executable_build_utils.h"],
    defines = ["JSV_ON_GCE"],
    visibility = [
        "//visibility:private",
        "@neptune//:__subpackages__",
    ],
    deps = [
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@llvm-project//mlir:IR",
        "@neptune//runtime/libs/ifrt/persistent_compilation_cache:ifrt_persistent_compilation_cache",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt/proto:compile_options_proto_cc",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/ifrt/hlo:hlo_program",
        "@xla//xla/python/pjrt_ifrt:xla_ifrt",
        "@xla//xla/service:compilation_environments",
        "@xla//xla/service:computation_placer_hdr",
        "@xla//xla/tsl/platform:statusor",
    ],
)
