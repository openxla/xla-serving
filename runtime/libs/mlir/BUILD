load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = [
    "//learning/infra/mira/experimental/orbax_model:__subpackages__",
    "@neptune//runtime:__subpackages__",
])

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/mlir/utils:type_util",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt:pjrt_dtype",
    ],
)

cc_library(
    name = "dump_mlir_util",
    srcs = ["dump_mlir_util.cc"],
    hdrs = ["dump_mlir_util.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@tsl//tsl/platform:path",
        "@xla//xla/tsl/lib/io:buffered_file",
        "@xla//xla/tsl/platform:env",
    ],
)
