load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

cc_binary(
    name = "jsv_client_main",
    srcs = ["client_main.cc"],
    deps = [
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
        "@neptune//runtime/serving/gcp/proto:prediction_log_cc_proto",
        "@neptune//runtime/serving/gcp/proto:prediction_service_cc_grpc_proto",
        "@neptune//runtime/tensorflow/utils:tf_to_jsv_proto",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:portable_gif_internal",
        "@xla//xla/python/ifrt:dtype_proto_cc",
        "@xla//xla/python/ifrt:shape_proto_cc",
    ],
)
