load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:py_proto_library.bzl", "py_proto_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(default_visibility = [
    "@xla_serving//data:neptune_data_internal",
])

proto_library(
    name = "resource_manager_proto",
    srcs = ["resource_manager.proto"],
    option_deps = ["@com_google_protobuf//:cpp_features_proto"],
)

py_proto_library(
    name = "resource_manager_py_pb2",
    deps = [":resource_manager_proto"],
)

cc_proto_library(
    name = "resource_manager_cc_proto",
    deps = [":resource_manager_proto"],
)
