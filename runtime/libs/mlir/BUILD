load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = [
    "@xla_serving//runtime:__subpackages__",
])

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
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
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/synchronization",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@tsl//tsl/platform:path",
        "@xla//xla/tsl/lib/io:buffered_file",
        "@xla//xla/tsl/platform:env",
    ],
)

cc_test(
    name = "dump_mlir_util_test",
    srcs = ["dump_mlir_util_test.cc"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS"],
    deps = [
        ":dump_mlir_util",
        "@googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@xla//xla/tsl/lib/core:status_test_util",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:test",
    ],
)
