load("//corp/cloud/image_from_binary:build_defs.bzl", "image_from_binary")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//third_party/bazel_rules/rules_docker/container:container.bzl", "container_push")

vardef("JSV_SERVING_REGISTRY", "gcr.io")

vardef("JSV_SERVING_REPOSITORY", "")

image_from_binary(
    name = "jsv_serving_image",
    binary = ":jsv_serving_main",
    exec_properties = {"mem": "28g"},  # Needed to avoid OOMs with compiling with -c dbg.
)

# To push a new image to GCR, run:
# blaze run --config=gce //third_party/neptune/runtime/serving/gcp/server:jsv_serving_pusher --define=REGISTRY=gcr.io --define=REPOSITORY=<gcp_project>/<repository_name>
container_push(
    name = "jsv_serving_pusher",
    format = "Docker",
    image = ":jsv_serving_image",
    registry = "$(JSV_SERVING_REGISTRY)",
    repository = "$(JSV_SERVING_REPOSITORY)",
    tags = ["latest"],
)

cc_binary(
    name = "jsv_serving_main",
    srcs = ["main.cc"],
    deps = [
        ":jsv_grpc_server",
        "//base",
        "@com_github_grpc_grpc//:grpc++",
        "@com_github_grpc_grpc//:grpc++_reflection",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@neptune//runtime/serving/gcp/auth:utils",
        "@xla//xla/tsl/platform:status",
    ],
)

cc_library(
    name = "jsv_grpc_server",
    srcs = ["orbax_grpc_server.cc"],
    hdrs = ["orbax_grpc_server.h"],
    deps = [
        "//learning/infra/mira/experimental/orbax_model/tensorflow:tf_compatible_interpreter_create_options",
        "//learning/infra/mira/experimental/orbax_model/tensorflow/utils:tf_to_host_buffer",
        "//third_party/tensorflow/core/framework:tensor",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@neptune//runtime/core:host_buffer",
        "@neptune//runtime/core:value",
        "@neptune//runtime/core/proto:host_buffer_cc_proto",
        "@neptune//runtime/interpreter",
        "@neptune//runtime/interpreter:run_options",
        "@neptune//runtime/kernel:kernel_io_data_type",
        "@neptune//runtime/serving/gcp/proto:get_model_metadata_cc_proto",
        "@neptune//runtime/serving/gcp/proto:model_cc_proto",
        "@neptune//runtime/serving/gcp/proto:predict_cc_proto",
        "@neptune//runtime/serving/gcp/proto:prediction_service_cc_grpc_proto",
        "@neptune//runtime/serving/gcp/proto:prediction_service_cc_proto",
        "@xla//xla/pjrt:pjrt_api",
        "@xla//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "@xla//xla/pjrt/plugin/xla_cpu:cpu_static_registration",
        "@xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "@xla//xla/pjrt/plugin/xla_gpu:gpu_static_registration",
        "@xla//xla/pjrt/plugin/xla_gpu:xla_gpu_client_options",
        "@xla//xla/pjrt/plugin/xla_gpu:xla_gpu_pjrt_client",
        "@xla//xla/pjrt/plugin/xla_tpu:xla_tpu_pjrt_client",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
)
