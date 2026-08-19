load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:py_proto_library.bzl", "py_proto_library")
load("@neptune//data/processor/python:build_defs.bzl", "transform_pybind_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "bucket_pad_transform_proto",
    srcs = ["bucket_pad_transform.proto"],
    deps = [
        "@neptune//data/processor/cc/utils:default_value_proto",
    ],
)

cc_proto_library(
    name = "bucket_pad_transform_cc_proto",
    deps = [":bucket_pad_transform_proto"],
)

py_proto_library(
    name = "bucket_pad_transform_py_pb2",
    deps = [":bucket_pad_transform_proto"],
)

cc_library(
    name = "bucket_pad_transform",
    srcs = ["bucket_pad_transform.cc"],
    hdrs = ["bucket_pad_transform.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        ":bucket_pad_transform_cc_proto",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
        "@neptune//data/processor/cc/api:registry",
        "@neptune//data/processor/cc/api:transform_context",
        "@neptune//data/processor/cc/api:type_info",
        "@neptune//data/processor/cc/core:transform",
        "@neptune//data/processor/cc/utils:default_value_cc_proto",
        "@neptune//data/processor/cc/utils:default_value_utils",
        "@neptune//data/processor/proto:transform_config_cc_proto",
        "@neptune//data/resource_manager/cc:resource",
        "@neptune//runtime/core:host_buffer",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/platform:status_macros",
    ],
    alwayslink = 1,
)

transform_pybind_library(
    name = "bucket_pad_transform_pybind",
    srcs = ["bucket_pad_transform.cc"],
    hdrs = ["bucket_pad_transform.h"],
    dynamic_deps = ["@neptune//data:neptune_data_core"],
    generate_wrappers = True,
    linkopts = ["-Wl,-rpath,$$ORIGIN/../.."],
    deps = [
        ":bucket_pad_transform",
    ],
)

cc_test(
    name = "bucket_pad_transform_test",
    srcs = ["bucket_pad_transform_test.cc"],
    defines = [
        "ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS",
        "ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS",
    ],
    deps = [
        ":bucket_pad_transform",
        ":bucket_pad_transform_cc_proto",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
        "@neptune//data/processor/cc/api:transform_context",
        "@neptune//data/processor/cc/core:transform",
        "@neptune//data/processor/cc/utils:default_value_cc_proto",
        "@neptune//data/processor/proto:transform_config_cc_proto",
        "@neptune//runtime/core:host_buffer",
        "@xla//xla/python/ifrt",
    ],
)
