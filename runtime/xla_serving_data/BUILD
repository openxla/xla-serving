load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

# Limit visibility to selected packages.
package_group(
    name = "composite_processor_function_internal_users",
    packages = [
        "//runtime/interpreter/...",
        "//runtime/python/...",
        "//runtime/serving/...",
    ],
)

cc_library(
    name = "composite_processor_op",
    srcs = ["composite_processor_op.cc"],
    hdrs = ["composite_processor_op.h"],
    deps = [
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/types:span",
        "@xla_serving//data/processor/cc/api:transform_context",
        "@xla_serving//data/processor/cc/api:value",
        "@xla_serving//data/processor/cc/core:composite_processor",
        "@xla_serving//runtime/core:host_buffer",
        "@xla_serving//runtime/core:value",
        "@xla_serving//runtime/kernel",
        "@xla_serving//runtime/kernel:kernel_io_data_type",
    ],
)

cc_library(
    name = "composite_processor_function_handler",
    srcs = ["composite_processor_function_handler.cc"],
    hdrs = ["composite_processor_function_handler.h"],
    visibility = [
        ":composite_processor_function_internal_users",
        "@xla_serving//:__subpackages__",
    ],
    deps = [
        ":composite_processor_op",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@tsl//tsl/platform:path",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
        "@xla_serving//data/processor/cc/core:composite_processor",
        "@xla_serving//data/processor/proto:transform_config_cc_proto",
        "@xla_serving//protos:manifest_cc_proto",
        "@xla_serving//protos:orchestration_cc_proto",
        "@xla_serving//runtime/interpreter:interpreter_create_options",
        "@xla_serving//runtime/interpreter:manifest_utils",
        "@xla_serving//runtime/interpreter:optional_function_handler",
        "@xla_serving//runtime/interpreter:optional_function_handler_factory",
        "@xla_serving//runtime/kernel",
        "@xla_serving//runtime/kernel:kernel_io_data_type",
    ],
    alwayslink = True,
)

cc_library(
    name = "interpreter_composite_processor_support",
    visibility = [
        ":composite_processor_function_internal_users",
        "@xla_serving//:__subpackages__",
    ],
    deps = [
        ":composite_processor_function_handler",
        "@xla_serving//data/kernels:numerical_transforms",
        "@xla_serving//data/kernels/bucket_pad:bucket_pad_transform",
        "@xla_serving//data/kernels/cpu_embedding:cpu_embedding_lookup_transform",
        "@xla_serving//data/kernels/cpu_embedding:pack_host_buffer_vector_transform",
        "@xla_serving//data/kernels/cpu_embedding:unpack_host_buffer_vector_transform",
    ],
)
