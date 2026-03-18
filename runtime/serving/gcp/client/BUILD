load("//corp/cloud/image_from_binary:build_defs.bzl", "image_from_binary")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("//third_party/bazel_rules/rules_docker/container:container.bzl", "container_push")

vardef("JSV_CLIENT_REGISTRY", "gcr.io")

vardef("JSV_CLIENT_REPOSITORY", "")

image_from_binary(
    name = "jsv_client_image",
    binary = ":jsv_client_main",
)

container_push(
    name = "jsv_cc_client_pusher",
    format = "Docker",
    image = ":jsv_client_image",
    registry = "$(JSV_CLIENT_REGISTRY)",
    repository = "$(JSV_CLIENT_REPOSITORY)",
    tags = ["latest"],
)

cc_binary(
    name = "jsv_client_main",
    srcs = ["client_main.cc"],
    deps = [
        "//base",
        "//learning/infra/mira/experimental/orbax_model/tensorflow/utils:tf_to_jsv_proto",
        "//third_party/tensorflow/core:lib",
        "//third_party/tensorflow/core:portable_gif_internal",
        "//third_party/tensorflow_serving/apis:prediction_log_cc_proto",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@com_github_grpc_grpc//:grpc++",
        "@neptune//runtime/core:host_buffer",
        "@neptune//runtime/core/proto:host_buffer_cc_proto",
        "@neptune//runtime/serving/gcp/auth:utils",
        "@neptune//runtime/serving/gcp/proto:model_cc_proto",
        "@neptune//runtime/serving/gcp/proto:predict_cc_proto",
        "@neptune//runtime/serving/gcp/proto:prediction_service_cc_grpc_proto",
        "@xla//xla/python/ifrt:dtype_proto_cc",
        "@xla//xla/python/ifrt:shape_proto_cc",
    ],
)
