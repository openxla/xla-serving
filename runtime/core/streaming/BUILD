load("@rules_cc//cc:cc_library.bzl", "cc_library")

package_group(
    name = "streaming_users",
    packages = [
        "//runtime/core/...",
        "//runtime/serving/google/streaming/...",
    ],
)

cc_library(
    name = "streaming",
    srcs = ["stream.cc"],
    hdrs = ["stream.h"],
    visibility = [":streaming_users"],
    deps = [
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/functional:any_invocable",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@neptune//runtime/core:host_buffer",
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
    ],
)
