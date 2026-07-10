load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@com_google_protobuf//bazel:py_proto_library.bzl", "py_proto_library")

package(default_visibility = [
    "//visibility:public",
])

# Proto definitions for host_buffer.proto
proto_library(
    name = "host_buffer_proto",
    srcs = ["host_buffer.proto"],
    deps = [
        "@xla//xla/python/ifrt:dtype_proto",
        "@xla//xla/python/ifrt:shape_proto",
    ],
)

cc_proto_library(
    name = "host_buffer_cc_proto",
    deps = [":host_buffer_proto"],
)

py_proto_library(
    name = "host_buffer_py_pb2",
    deps = [":host_buffer_proto"],
)

# Proto definitions for get_model_metadata.proto
proto_library(
    name = "get_model_metadata_proto",
    srcs = ["get_model_metadata.proto"],
    deps = [
        ":model_proto",
        "@com_google_protobuf//:any_proto",
        "@neptune//protos:orchestration_proto",
        "@neptune//protos:type_proto",
    ],
)

cc_proto_library(
    name = "get_model_metadata_cc_proto",
    deps = [":get_model_metadata_proto"],
)

# Proto definitions for model.proto
proto_library(
    name = "model_proto",
    srcs = ["model.proto"],
)

cc_proto_library(
    name = "model_cc_proto",
    deps = [":model_proto"],
)

py_proto_library(
    name = "model_py_pb2",
    deps = [":model_proto"],
)

# Proto definitions for predict.proto
proto_library(
    name = "predict_proto",
    srcs = ["predict.proto"],
    deps = [
        ":host_buffer_proto",
        ":model_proto",
    ],
)

cc_proto_library(
    name = "predict_cc_proto",
    deps = [":predict_proto"],
)

py_proto_library(
    name = "predict_py_pb2",
    deps = [":predict_proto"],
)

# Proto definitions for prediction_service.proto
proto_library(
    name = "prediction_service_proto",
    srcs = ["prediction_service.proto"],
    deps = [
        ":get_model_metadata_proto",
        ":predict_proto",
    ],
)

proto_library(
    name = "prediction_log_proto",
    srcs = ["prediction_log.proto"],
    deps = [
        ":model_proto",
        ":predict_proto",
    ],
)

cc_proto_library(
    name = "prediction_log_cc_proto",
    deps = [":prediction_log_proto"],
)

py_proto_library(
    name = "prediction_log_py_pb2",
    deps = [":prediction_log_proto"],
)

cc_proto_library(
    name = "prediction_service_cc_proto",
    deps = [":prediction_service_proto"],
)

alias(
    name = "prediction_service_cc_proto_only",
    actual = ":prediction_service_proto",
)

cc_grpc_library(
    name = "prediction_service_cc_grpc_proto",
    srcs = [":prediction_service_proto"],
    grpc_only = True,
    deps = [":prediction_service_cc_proto"],
)
