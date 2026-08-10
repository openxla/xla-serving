load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package_group(
    name = "kernel_users",
    packages = [
        "//runtime/tensorflow/...",
    ],
)

cc_library(
    name = "builtin_tensorflow_kernel",
    srcs = ["builtin_tensorflow_kernel.cc"],
    hdrs = ["builtin_tensorflow_kernel.h"],
    deps = [
        ":tf_saved_model_function_runner_op",
        "@abseil-cpp//absl/status",
        "@neptune//runtime/kernel",
        "@xla//xla/tsl/platform:errors",
    ],
)

cc_library(
    name = "init_lib_tensorflow",
    srcs = ["init_lib_tensorflow.cc"],
    visibility = [":kernel_users"],
    deps = [
        ":builtin_tensorflow_kernel",
        "@abseil-cpp//absl/log:check",
        "@neptune//runtime/kernel",
    ],
    alwayslink = 1,
)

cc_library(
    name = "kernel_builder_constants",
    hdrs = ["kernel_builder_constants.h"],
    visibility = [":kernel_users"],
    deps = ["@abseil-cpp//absl/strings:string_view"],
)

cc_library(
    name = "tf_saved_model_function_runner_op",
    srcs = ["tf_saved_model_function_runner_op.cc"],
    hdrs = ["tf_saved_model_function_runner_op.h"],
    deps = [
        ":kernel_builder_constants",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/time",
        "@abseil-cpp//absl/types:span",
        "@neptune//runtime/core:flat_signature",
        "@neptune//runtime/core:host_buffer",
        "@neptune//runtime/core:value",
        "@neptune//runtime/core:value_utils",
        "@neptune//runtime/kernel",
        "@neptune//runtime/kernel:kernel_builder_context",
        "@neptune//runtime/kernel:kernel_io_data_type",
        "@neptune//runtime/tensorflow:tf_saved_model_runner",
        "@neptune//runtime/tensorflow/utils:tf_to_host_buffer",
        "@org_tensorflow//tensorflow/core:framework",
        "@tsl//tsl/profiler/lib:traceme",
        "@tsl//tsl/profiler/lib:traceme_encode",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
)
