load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package_group(
    name = "kernel_users",
    packages = [
        "//learning/infra/mira/experimental/orbax_model/tensorflow/...",
        "@neptune//runtime/interpreter/...",
    ],
)

package_group(
    name = "kernel_internal",
    packages = [
        "//learning/infra/mira/experimental/orbax_model/tensorflow/kernel/...",
        "@neptune//runtime/kernel/...",
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
    ],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
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
        "@com_google_absl//absl/status",
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
    ],
    deps = [
        ":kernel_builder_context",
        ":kernel_io_data_type",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@neptune//runtime/core:value",
        "@neptune//runtime/interpreter:context",
        "@xla//xla/pjrt:pjrt_future",
        "@xla//xla/tsl/concurrency:future",
    ],
)

cc_library(
    name = "stablehlo_runner_op",
    srcs = ["stablehlo_runner_op.cc"],
    hdrs = ["stablehlo_runner_op.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":kernel",
        ":kernel_builder_context",
        ":kernel_io_data_type",
        "//third_party/protobuf",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@neptune//runtime/core:flat_signature",
        "@neptune//runtime/core:host_buffer",
        "@neptune//runtime/core:ifrt_loaded_variable_registry",
        "@neptune//runtime/core:stablehlo",
        "@neptune//runtime/core:stablehlo_optimizer",
        "@neptune//runtime/core:stablehlo_runner",
        "@neptune//runtime/core:value_utils",
        "@neptune//runtime/interpreter:manifest_utils",
        "@neptune//runtime/libs/ifrt/persistent_compilation_cache:ifrt_persistent_compilation_cache",
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
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "test_utils",
    testonly = True,
    hdrs = ["test_utils.h"],
    visibility = [":kernel_internal"],
    deps = [
        ":kernel",
        ":kernel_builder_context",
        ":stablehlo_runner_op",
        "//devtools/build/runtime:get_runfiles_dir",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@neptune//runtime/core:host_buffer",
        "@neptune//runtime/core:ifrt_loaded_variable_registry",
        "@neptune//runtime/core:sharding_utils",
        "@neptune//runtime/core:stablehlo",
        "@neptune//runtime/core:stablehlo_optimizer",
        "@neptune//runtime/core:value",
        "@neptune//runtime/interpreter:manifest_utils",
        "@neptune//runtime/libs/ifrt/persistent_compilation_cache:ifrt_persistent_compilation_cache",
        "@tsl//tsl/platform:protobuf",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/ir:hlo_sharding",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/ifrt:test_util",
        "@xla//xla/python/pjrt_ifrt:tfrt_cpu_client_test_lib",
        "@xla//xla/python/pjrt_ifrt:xla_ifrt",
        "@xla//xla/tsl/concurrency:future",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:status",
        "@xla//xla/tsl/platform:statusor",
    ],
)
