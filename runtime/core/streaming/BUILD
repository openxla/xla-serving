load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package_group(
    name = "streaming_users",
    packages = [
        "//runtime/core/...",
        "//runtime/interpreter/...",
        "//runtime/kernel/...",
        "//runtime/serving/google/streaming/...",
    ],
)

cc_library(
    name = "streaming",
    srcs = ["stream.cc"],
    hdrs = ["stream.h"],
    visibility = [":streaming_users"],
    deps = [
        ":utils",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/functional:any_invocable",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@abseil-cpp//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@stablehlo//:stablehlo_ops",
        "@tsl//tsl/platform:random",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//xla:executable_run_options",
        "@xla//xla/ffi",
        "@xla//xla/ffi:ffi_api",  # buildcleaner: keep
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt:pjrt_dtype",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:threadpool_interface",
        "@xla_serving//runtime/core:host_buffer",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    visibility = [":streaming_users"],
    deps = [
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:vlog_is_on",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
        "@xla_serving//runtime/core:host_buffer",
        "@xla_serving//runtime/libs/mlir:dump_mlir_util",
    ],
)
