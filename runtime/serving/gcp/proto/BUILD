load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_cc//cc:defs.bzl", "cc_proto_library")

package(default_visibility = [
    "//learning/infra/mira/experimental/orbax_model:__subpackages__",
    "@neptune//runtime:__subpackages__",
])

# Proto definitions for get_model_metadata.proto
proto_library(
    name = "get_model_metadata_proto",
    srcs = ["get_model_metadata.proto"],
    deps = [
        ":model_proto",
        "@com_google_protobuf//:protobuf", # any
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

# Proto definitions for predict.proto
proto_library(
    name = "predict_proto",
    srcs = ["predict.proto"],
    deps = [
        ":model_proto",
        "@neptune//runtime/core/proto:host_buffer_proto",
    ],
)

cc_proto_library(
    name = "predict_cc_proto",
    deps = [":predict_proto"],
)

# Proto definitions for prediction_service.proto
proto_library(
    name = "prediction_service_proto",
    srcs = ["prediction_service.proto"],
    has_services = True,
    deps = [
        ":get_model_metadata_proto",
        ":predict_proto",
    ],
)

cc_proto_library(
    name = "prediction_service_cc_proto",
    deps = [":prediction_service_proto"],
)

cc_grpc_library(
    name = "prediction_service_cc_grpc_proto",
    srcs = [":prediction_service_proto"],
    allow_cq_api = True,
    allow_sync_server_api = True,
    deps = [":prediction_service_cc_proto"],
)
