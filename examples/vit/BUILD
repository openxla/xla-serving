load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

package(default_visibility = [
    "@xla_serving//examples:__subpackages__",
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
        "@xla//xla/python/ifrt:dtype_proto_cc",
        "@xla//xla/python/ifrt:shape_proto_cc",
        "@xla_serving//protos/grpc:host_buffer_cc_proto",
        "@xla_serving//protos/grpc:model_cc_proto",
        "@xla_serving//protos/grpc:predict_cc_proto",
        "@xla_serving//protos/grpc:prediction_service_cc_grpc_proto",
        "@xla_serving//runtime/serving/gcp/auth:utils",
    ],
)
