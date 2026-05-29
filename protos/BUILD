load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:py_proto_library.bzl", "py_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "type_proto",
    srcs = ["type.proto"],
    deps = [
        "@xla//xla:xla_data_proto",
    ],
)

cc_proto_library(
    name = "type_cc_proto",
    deps = [":type_proto"],
)

py_proto_library(
    name = "type_py_pb2",
    deps = [":type_proto"],
)

proto_library(
    name = "manifest_proto",
    srcs = ["manifest.proto"],
    deps = [
        ":type_proto",
        "@xla//xla/pjrt/proto:compile_options_proto",
    ],
)

cc_proto_library(
    name = "manifest_cc_proto",
    deps = [":manifest_proto"],
)

py_proto_library(
    name = "manifest_py_pb2",
    deps = [":manifest_proto"],
)

proto_library(
    name = "orchestration_proto",
    srcs = ["orchestration.proto"],
    deps = [
        ":type_proto",
    ],
)

cc_proto_library(
    name = "orchestration_cc_proto",
    deps = [":orchestration_proto"],
)

py_proto_library(
    name = "orchestration_py_pb2",
    deps = [":orchestration_proto"],
)

proto_library(
    name = "tf_concrete_function_handle_proto",
    srcs = ["tf_concrete_function_handle.proto"],
)

cc_proto_library(
    name = "tf_concrete_function_handle_cc_proto",
    deps = [":tf_concrete_function_handle_proto"],
)

py_proto_library(
    name = "tf_concrete_function_handle_py_pb2",
    deps = [":tf_concrete_function_handle_proto"],
)

proto_library(
    name = "asset_map_proto",
    srcs = ["asset_map.proto"],
)

cc_proto_library(
    name = "asset_map_cc_proto",
    deps = [":asset_map_proto"],
)

py_proto_library(
    name = "asset_map_py_pb2",
    deps = [":asset_map_proto"],
)
