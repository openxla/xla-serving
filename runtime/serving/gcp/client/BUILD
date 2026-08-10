load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
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
        "@grpc//:grpc++",
        "@neptune//protos/grpc:host_buffer_cc_proto",
        "@neptune//protos/grpc:model_cc_proto",
        "@neptune//protos/grpc:predict_cc_proto",
        "@neptune//protos/grpc:prediction_log_cc_proto",
        "@neptune//protos/grpc:prediction_service_cc_grpc_proto",
        "@neptune//runtime/core:host_buffer",
        "@neptune//runtime/serving/gcp/auth:utils",
        "@neptune//runtime/tensorflow/utils:tf_to_jsv_proto",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:portable_gif_internal",
        "@xla//xla/python/ifrt:dtype_proto_cc",
        "@xla//xla/python/ifrt:shape_proto_cc",
    ],
)
