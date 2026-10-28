load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = [
    "@xla_serving//runtime/python:__subpackages__",
    "@xla_serving//runtime/serving:__subpackages__",
    "@xla_serving//runtime/tensorflow:__subpackages__",
])

cc_library(
    name = "common_utils",
    srcs = ["common_utils.cc"],
    hdrs = ["common_utils.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
        "@xla_serving//protos:manifest_cc_proto",
        "@xla_serving//runtime/core:common_util",
        "@xla_serving//runtime/core:constants",
        "@xla_serving//runtime/interpreter:manifest_utils",
    ],
)

cc_library(
    name = "ifrt_tensor_utils",
    srcs = ["ifrt_tensor_utils.cc"],
    hdrs = ["ifrt_tensor_utils.h"],
    deps = [
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@xla//xla:shape_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt:pjrt_dtype",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_test(
    name = "ifrt_tensor_utils_test",
    srcs = ["ifrt_tensor_utils_test.cc"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS"],
    deps = [
        ":ifrt_tensor_utils",
        "@abseil-cpp//absl/status:status_matchers",
        "@googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/python/ifrt",
    ],
)

cc_library(
    name = "tf_to_host_buffer",
    srcs = ["tf_to_host_buffer.cc"],
    hdrs = ["tf_to_host_buffer.h"],
    deps = [
        ":ifrt_tensor_utils",
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@tsl//tsl/platform:refcount",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/platform:statusor",
        "@xla_serving//runtime/core:host_buffer",
        "@xla_serving//runtime/core:value",
        "@xla_serving//runtime/kernel:kernel_io_data_type",
    ],
)

# Registers a `tensorflow::Tensor` -> `HostBuffer` converter in the global
# `HostBufferConverterRegistry`, so StableHLO runners accept `Tensor` inputs.
cc_library(
    name = "tf_host_buffer_converter",
    srcs = ["tf_host_buffer_converter.cc"],
    hdrs = ["tf_host_buffer_converter.h"],
    deps = [
        ":tf_to_host_buffer",
        "@abseil-cpp//absl/status:statusor",
        "@org_tensorflow//tensorflow/core:framework",
        "@xla_serving//runtime/core:host_buffer",
        "@xla_serving//runtime/core:host_buffer_converter",
    ],
    alwayslink = True,
)

cc_test(
    name = "tf_host_buffer_converter_test",
    srcs = ["tf_host_buffer_converter_test.cc"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS"],
    deps = [
        ":tf_host_buffer_converter",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:testlib",
        "@xla//xla/python/ifrt",
        "@xla_serving//runtime/core:host_buffer",
        "@xla_serving//runtime/core:host_buffer_converter",
        "@xla_serving//runtime/core:host_buffer_matcher",
        "@xla_serving//runtime/core:host_buffer_test_utils",
        "@xla_serving//runtime/core:value",
    ],
)

cc_test(
    name = "tf_to_host_buffer_test",
    srcs = ["tf_to_host_buffer_test.cc"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS"],
    deps = [
        ":tf_to_host_buffer",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:portable_gif_internal",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:test",
        "@org_tensorflow//tensorflow/core:testlib",
        "@org_tensorflow//tensorflow/core/framework:tensor_matcher",
        "@tsl//tsl/platform:ml_dtypes",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:types",
        "@xla_serving//runtime/core:default_allocator",
        "@xla_serving//runtime/core:host_buffer",
        "@xla_serving//runtime/core:host_buffer_matcher",
        "@xla_serving//runtime/core:host_buffer_test_utils",
        "@xla_serving//runtime/core:value",
        "@xla_serving//runtime/kernel:kernel_io_data_type",
    ],
)

cc_test(
    name = "tf_to_host_buffer_proto_test",
    srcs = ["tf_to_host_buffer_proto_test.cc"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS"],
    deps = [
        ":tf_to_host_buffer_proto",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/framework:tensor",
        "@org_tensorflow//tensorflow/core/framework:tensor_shape",
        "@tsl//tsl/platform:tstring",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/util/proto:proto_matchers",
        "@xla_serving//protos/grpc:host_buffer_cc_proto",
        "@xla_serving//protos/grpc:model_cc_proto",
        "@xla_serving//protos/grpc:predict_cc_proto",
        "@xla_serving//runtime/core:host_buffer",
    ],
)

cc_library(
    name = "tf_to_host_buffer_proto",
    srcs = ["tf_to_host_buffer_proto.cc"],
    hdrs = ["tf_to_host_buffer_proto.h"],
    deps = [
        ":ifrt_tensor_utils",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/framework:tensor",
        "@tsl//tsl/platform:tstring",
        "@xla//xla/python/ifrt:dtype_proto_cc",
        "@xla//xla/tsl/platform:statusor",
        "@xla_serving//protos/grpc:host_buffer_cc_proto",
        "@xla_serving//protos/grpc:model_cc_proto",
        "@xla_serving//protos/grpc:predict_cc_proto",
    ],
)
