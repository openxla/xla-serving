load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:py_proto_library.bzl", "py_proto_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(default_visibility = ["@xla_serving//:__subpackages__"])

exports_files(["scan_ops.cc"] + glob([
    "op_*.cc",
    "reduction_ops_*.cc",
    "*.h",
]))

filegroup(
    name = "math_ops_srcs",
    srcs = ["scan_ops.cc"] + glob(
        [
            "op_*.cc",
            "reduction_ops_*.cc",
        ],
        exclude = ["op_common_test.cc"],
    ),
)

proto_library(
    name = "reduction_transform_config_proto",
    srcs = ["reduction_transform_config.proto"],
)

py_proto_library(
    name = "reduction_transform_config_py_pb2",
    deps = [":reduction_transform_config_proto"],
)

cc_proto_library(
    name = "reduction_transform_config_cc_proto",
    deps = [":reduction_transform_config_proto"],
)

proto_library(
    name = "scan_transform_config_proto",
    srcs = ["scan_transform_config.proto"],
)

py_proto_library(
    name = "scan_transform_config_py_pb2",
    deps = [":scan_transform_config_proto"],
)

cc_proto_library(
    name = "scan_transform_config_cc_proto",
    deps = [":scan_transform_config_proto"],
)

cc_library(
    name = "op_functors",
    hdrs = [
        "op_functors.h",
        "special_math_impl.h",
    ],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@eigen_archive//:eigen3",
        "@tsl//tsl/platform:tstring",
        "@xla_serving//data/kernels:bounds_check",
    ],
)

cc_test(
    name = "op_common_test",
    srcs = ["op_common_test.cc"],
    defines = [
        "ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS",
        "ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS",
    ],
    deps = [
        ":op_functors",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@googletest//:gtest_main",
        "@tsl//tsl/platform:tstring",
        "@xla//xla/python/ifrt",
        "@xla_serving//data/kernels:op_common",
        "@xla_serving//data/kernels:test_utils",
        "@xla_serving//data/kernels/bitwise:op_functors",
        "@xla_serving//data/processor/cc/api:transform_context",
        "@xla_serving//data/test_utils:status_macros",
        "@xla_serving//runtime/core:host_buffer",
    ],
)
