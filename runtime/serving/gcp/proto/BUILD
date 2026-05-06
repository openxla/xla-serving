load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(default_visibility = [
    "//learning/infra/mira/experimental/orbax_model:__subpackages__",
    "@neptune//examples:__subpackages__",
    "@neptune//runtime:__subpackages__",
])

# Proto definitions for get_model_metadata.proto
proto_library(
    name = "get_model_metadata_proto",
    srcs = ["get_model_metadata.proto"],
    import_prefix = "neptune_model",
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
    import_prefix = "neptune_model",
)

cc_proto_library(
    name = "model_cc_proto",
    deps = [":model_proto"],
)

# Proto definitions for predict.proto
proto_library(
    name = "predict_proto",
    srcs = ["predict.proto"],
    import_prefix = "neptune_model",
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
    import_prefix = "neptune_model",
    deps = [
        ":get_model_metadata_proto",
        ":predict_proto",
    ],
)

proto_library(
    name = "prediction_log_proto",
    srcs = ["prediction_log.proto"],
    import_prefix = "neptune_model",
    deps = [
        ":model_proto",
        ":predict_proto",
    ],
)

cc_proto_library(
    name = "prediction_log_cc_proto",
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
