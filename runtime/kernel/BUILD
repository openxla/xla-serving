load("@rules_cc//cc:cc_library.bzl", "cc_library")

package_group(
    name = "kernel_users",
    packages = [
        "//runtime/interpreter/...",
        "//runtime/tensorflow/...",
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
    includes = ["@neptune//runtime/interpreter:interpreter_users"],
)

filegroup(
    name = "testdata",
    srcs = glob([
        "testdata/*.mlir",
    ]),
    visibility = [":kernel_internal"],
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
        "@neptune//runtime/tensorflow:__subpackages__",
    ],
    deps = [
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@neptune//runtime/core:value",
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
        "@neptune//runtime/kernel/batching:batch_runner_op",
        "@xla//xla/tsl/platform:errors",
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
        "@neptune//runtime/tensorflow:__subpackages__",
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
        "@neptune//runtime/core:flat_signature",
        "@neptune//runtime/core:value",
        "@neptune//runtime/interpreter:context",
        "@xla//xla/pjrt:pjrt_future",
        "@xla//xla/tsl/concurrency:future",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
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
        "@com_google_protobuf//:protobuf",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@neptune//runtime/core:flat_signature",
        "@neptune//runtime/core:ifrt_loaded_variable_registry",
        "@neptune//runtime/core:stablehlo",
        "@neptune//runtime/core:stablehlo_optimizer",
        "@neptune//runtime/core:stablehlo_runner",
        "@neptune//runtime/core:value",
        "@neptune//runtime/interpreter:manifest_utils",
        "@neptune//runtime/libs/ifrt/persistent_compilation_cache:ifrt_persistent_compilation_cache",
        "@neptune//runtime/libs/ifrt/serving_core_selector",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/pjrt/proto:compile_options_proto_cc",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:statusor",
    ],
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
