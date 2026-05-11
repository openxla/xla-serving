load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@com_google_protobuf//bazel:py_proto_library.bzl", "py_proto_library")

package(default_visibility = [
    "@neptune//neptune_model:__subpackages__",
])

proto_library(
    name = "xla_vendored_proto",
    srcs = [
        "autotune_results.proto",
        "autotuning.proto",
        "backends/autotuner/backends.proto",
        "pjrt/proto/compile_options.proto",
        "service/hlo.proto",
        "service/metrics.proto",
        "stream_executor/cuda/cuda_compute_capability.proto",
        "stream_executor/device_description.proto",
        "stream_executor/sycl/oneapi_compute_capability.proto",
        "tsl/protobuf/dnn.proto",
        "xla.proto",
        "xla_data.proto",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_protobuf//:any_proto",
        "@com_google_protobuf//:duration_proto",
        "@com_google_protobuf//:timestamp_proto",
        "@com_google_protobuf//:wrappers_proto",
    ],
)

cc_proto_library(
    name = "xla_vendored_cc_proto",
    visibility = ["//visibility:public"],
    deps = [":xla_vendored_proto"],
)

py_proto_library(
    name = "xla_vendored_py_pb2",
    visibility = ["//visibility:public"],
    deps = [":xla_vendored_proto"],
)
