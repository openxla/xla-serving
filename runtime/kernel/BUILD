load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package_group(
    name = "kernel_users",
    packages = [
        "//runtime/interpreter/...",
        "//runtime/tensorflow/...",
        "//runtime/xla_serving_data/...",
    ],
)

package_group(
    name = "kernel_internal",
    packages = [
        "//runtime/kernel/...",
    ],
)

# Allow users of interpreter to use `KernelIODataType`.
package_group(
    name = "interpreter_users",
    includes = ["@xla_serving//runtime/interpreter:interpreter_users"],
)

cc_library(
    name = "kernel_builder_context",
    srcs = [],
    hdrs = [
        "kernel_builder_constants.h",
        "kernel_builder_context.h",
    ],
    visibility = [
        ":kernel_internal",
        ":kernel_users",
        "@xla_serving//runtime/tensorflow:__subpackages__",
    ],
    deps = [
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@xla_serving//runtime/core:value",
    ],
)

cc_library(
    name = "builtin_kernel",
    srcs = ["builtin_kernel.cc"],
    hdrs = ["builtin_kernel.h"],
    visibility = [":kernel_users"],
    deps = [
        ":identity",
        ":kernel",
        ":stablehlo_runner_op",
        "@abseil-cpp//absl/status",
        "@xla//xla/tsl/platform:errors",
        "@xla_serving//runtime/kernel/batching:batch_runner_op",
    ],
)

cc_library(
    name = "kernel_io_data_type",
    hdrs = ["kernel_io_data_type.h"],
    visibility = [
        ":interpreter_users",
        ":kernel_internal",
        ":kernel_users",
    ],
)

cc_library(
    name = "kernel",
    srcs = ["kernel.cc"],
    hdrs = ["kernel.h"],
    visibility = [
        ":kernel_internal",
        ":kernel_users",
        "@xla_serving//runtime/tensorflow:__subpackages__",
    ],
    deps = [
        ":kernel_builder_context",
        ":kernel_io_data_type",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/functional:any_invocable",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/types:span",
        "@xla//xla/pjrt:pjrt_future",
        "@xla//xla/tsl/concurrency:future",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
        "@xla_serving//runtime/core:flat_signature",
        "@xla_serving//runtime/core:value",
        "@xla_serving//runtime/interpreter:context",
    ],
)

cc_library(
    name = "stablehlo_runner_op",
    srcs = ["stablehlo_runner_op.cc"],
    hdrs = ["stablehlo_runner_op.h"],
    visibility = [":kernel_internal"],
    deps = [
        ":kernel",
        ":kernel_builder_context",
        ":kernel_io_data_type",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/functional:any_invocable",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/types:span",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:statusor",
        "@xla_serving//runtime/core:flat_signature",
        "@xla_serving//runtime/core:host_buffer",
        "@xla_serving//runtime/core:ifrt_loaded_variable_registry",
        "@xla_serving//runtime/core:stablehlo",
        "@xla_serving//runtime/core:stablehlo_runner_base",
        "@xla_serving//runtime/core:value",
        "@xla_serving//runtime/core/streaming:utils",
        "@xla_serving//runtime/interpreter:manifest_utils",
    ],
)

cc_library(
    name = "aot_stablehlo_runner_registration",
    srcs = ["aot_stablehlo_runner_registration.cc"],
    visibility = [
        ":kernel_internal",
        ":kernel_users",
    ],
    deps = [
        ":kernel_builder_context",
        ":stablehlo_runner_op",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:statusor",
        "@xla_serving//runtime/core:aot_stablehlo_runner",
        "@xla_serving//runtime/core:stablehlo",
        "@xla_serving//runtime/core:stablehlo_runner_base",
    ],
    alwayslink = 1,
)

cc_library(
    name = "jit_stablehlo_runner_registration",
    srcs = ["jit_stablehlo_runner_registration.cc"],
    visibility = [
        ":kernel_internal",
        ":kernel_users",
    ],
    deps = [
        ":kernel_builder_context",
        ":stablehlo_runner_op",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@com_google_protobuf//:protobuf",
        "@xla//xla/pjrt/proto:compile_options_proto_cc",
        "@xla//xla/tsl/platform:statusor",
        "@xla_serving//runtime/core:stablehlo",
        "@xla_serving//runtime/core:stablehlo_optimizer",
        "@xla_serving//runtime/core:stablehlo_runner",
        "@xla_serving//runtime/core:stablehlo_runner_base",
        "@xla_serving//runtime/libs/ifrt/persistent_compilation_cache:ifrt_persistent_compilation_cache",
        "@xla_serving//runtime/libs/ifrt/serving_core_selector",
    ],
    alwayslink = 1,
)

cc_library(
    name = "identity",
    srcs = ["identity.cc"],
    hdrs = ["identity.h"],
    visibility = [":kernel_internal"],  # For testing.
    deps = [
        ":kernel",
        ":kernel_builder_context",
        ":kernel_io_data_type",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@xla//xla/tsl/platform:statusor",
    ],
)
