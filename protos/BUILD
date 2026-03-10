load("@neptune//protos:build_defs.bzl", "ml_proto_library")
load("//tools/build_defs/testing:bzl_library.bzl", "bzl_library")

package(default_visibility = [
    # TODO b/361346745 - Remove/update this visibility once the prototpying is done.
    "//learning/infra/mira/experimental/orbax_model:__subpackages__",
    "//learning/serving/servables/tfrt:__subpackages__",
    "//smartass/brain/lemming:__subpackages__",
    "//third_party/py/orbax/experimental/model/cli:__subpackages__",
    "//third_party/py/orbax/experimental/model/core:__subpackages__",
    "//third_party/py/orbax/experimental/model/test_utils:__subpackages__",
    "//third_party/py/orbax/export:__subpackages__",
    "@neptune//:__subpackages__",
    "//learning/tfx/pipeline/infra_validation:__subpackages__",
])

ml_proto_library(
    name = "type_proto",
    srcs = ["type.proto"],
    deps = [
        "@xla//xla:xla_data_proto",
    ],
)

ml_proto_library(
    name = "manifest_proto",
    srcs = ["manifest.proto"],
    deps = [
        ":type_proto",
        "@xla//xla/pjrt/proto:compile_options_proto",
    ],
)

ml_proto_library(
    name = "orchestration_proto",
    srcs = ["orchestration.proto"],
    deps = [
        ":type_proto",
    ],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    deps = [
        "//devtools/build_cleaner/skylark:build_defs_lib",
        "//third_party/bazel_rules/rules_python/python:proto_bzl",
        "@com_google_protobuf//bazel:proto_library_bzl",
        "//tools/build_defs/proto/cpp:cc_proto_library_bzl",
    ],
)
