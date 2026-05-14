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
        "@com_github_grpc_grpc//:grpc++",
        "@neptune//runtime/core:host_buffer",
        "@neptune//runtime/core/proto:host_buffer_cc_proto",
        "@neptune//runtime/serving/gcp/auth:utils",
        "@neptune//runtime/serving/gcp/proto:model_cc_proto",
        "@neptune//runtime/serving/gcp/proto:predict_cc_proto",
        "@neptune//runtime/serving/gcp/proto:prediction_log_cc_proto",
        "@neptune//runtime/serving/gcp/proto:prediction_service_cc_grpc_proto",
        "@xla//xla/python/ifrt:dtype_proto_cc",
        "@xla//xla/python/ifrt:shape_proto_cc",
        "@xla//xla/tsl/platform:statusor",
    ],
)
