load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

package(default_visibility = [
    "@neptune//examples:__subpackages__",
])

cc_binary(
    name = "client",
    srcs = ["client.cc"],
    deps = [
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/flags:parse",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/time",
        "@grpc//:grpc++",
        "@neptune//protos/grpc:host_buffer_cc_proto",
        "@neptune//protos/grpc:model_cc_proto",
        "@neptune//protos/grpc:predict_cc_proto",
        "@neptune//protos/grpc:prediction_service_cc_grpc_proto",
        "@neptune//runtime/serving/gcp/auth:utils",
        "@xla//xla/python/ifrt:dtype_proto_cc",
        "@xla//xla/python/ifrt:shape_proto_cc",
    ],
)
