load("@rules_proto//proto:defs.bzl", "proto_library")
load("//tools/build_defs/proto/cpp:cc_proto_library.bzl", "cc_proto_library")

package(default_visibility = [
    "//learning/infra/mira/experimental/orbax_model:__subpackages__",
    "@neptune//runtime:__subpackages__",
])

proto_library(
    name = "host_buffer_proto",
    srcs = ["host_buffer.proto"],
    deps = [
        "@xla//xla/python/ifrt:dtype_proto",
        "@xla//xla/python/ifrt:shape_proto",
    ],
)

cc_proto_library(
    name = "host_buffer_cc_proto",
    deps = [":host_buffer_proto"],
)
