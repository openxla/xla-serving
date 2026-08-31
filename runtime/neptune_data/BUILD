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
        "@neptune//data/processor/cc/api:transform_context",
        "@neptune//data/processor/cc/api:value",
        "@neptune//data/processor/cc/core:composite_processor",
        "@neptune//runtime/core:host_buffer",
        "@neptune//runtime/core:value",
        "@neptune//runtime/kernel",
        "@neptune//runtime/kernel:kernel_io_data_type",
    ],
)

cc_library(
    name = "composite_processor_function_handler",
    srcs = ["composite_processor_function_handler.cc"],
    hdrs = ["composite_processor_function_handler.h"],
    visibility = [":composite_processor_function_internal_users"],
    deps = [
        ":composite_processor_op",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@neptune//data/processor/cc/core:composite_processor",
        "@neptune//data/processor/proto:transform_config_cc_proto",
        "@neptune//protos:manifest_cc_proto",
        "@neptune//protos:orchestration_cc_proto",
        "@neptune//runtime/interpreter:interpreter_create_options",
        "@neptune//runtime/interpreter:manifest_utils",
        "@neptune//runtime/interpreter:optional_function_handler",
        "@neptune//runtime/interpreter:optional_function_handler_factory",
        "@neptune//runtime/kernel",
        "@neptune//runtime/kernel:kernel_io_data_type",
        "@tsl//tsl/platform:path",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
    alwayslink = True,
)

# Include this library in the `deps` of the `cc_binary` together with `//third_party/neptune/runtime/interpreter` to enable JAX Data-based preprocessing and postprocessing.
cc_library(
    name = "interpreter_composite_processor_support",
    visibility = [
        ":composite_processor_function_internal_users",
        "//third_party/py/orbax/export:__subpackages__",
        "@neptune//:__subpackages__",
    ],
    deps = [
        ":composite_processor_function_handler",
        "@neptune//data/kernels/cpu_embedding:cpu_embedding_lookup_transform",
        "@neptune//data/kernels/cpu_embedding:pack_host_buffer_vector_transform",
        "@neptune//data/kernels/cpu_embedding:unpack_host_buffer_vector_transform",
    ],
)
