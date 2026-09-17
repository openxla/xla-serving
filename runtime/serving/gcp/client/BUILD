load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

cc_binary(
    name = "xla_serving_client_main",
    srcs = ["client_main.cc"],
    deps = [
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@grpc//:grpc++",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:portable_gif_internal",
        "@xla//xla/python/ifrt:dtype_proto_cc",
        "@xla//xla/python/ifrt:shape_proto_cc",
        "@xla_serving//protos/grpc:host_buffer_cc_proto",
        "@xla_serving//protos/grpc:model_cc_proto",
        "@xla_serving//protos/grpc:predict_cc_proto",
        "@xla_serving//protos/grpc:prediction_log_cc_proto",
        "@xla_serving//protos/grpc:prediction_service_cc_grpc_proto",
        "@xla_serving//runtime/core:host_buffer",
        "@xla_serving//runtime/serving/gcp/auth:utils",
        "@xla_serving//runtime/tensorflow/utils:tf_to_host_buffer_proto",
    ],
)
