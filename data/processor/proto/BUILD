load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:py_proto_library.bzl", "py_proto_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(default_visibility = ["@xla_serving//data:xla_serving_data_internal"])

proto_library(
    name = "transform_config_proto",
    srcs = ["transform_config.proto"],
    deps = [
        "@com_google_protobuf//:any_proto",
        "@xla_serving//data/resource_manager/proto:resource_manager_proto",
    ],
)

py_proto_library(
    name = "transform_config_py_pb2",
    visibility = [
        "@xla_serving//data:xla_serving_data_internal",
    ],
    deps = [":transform_config_proto"],
)

cc_proto_library(
    name = "transform_config_cc_proto",
    deps = [":transform_config_proto"],
)
