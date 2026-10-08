load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["@xla_serving//:__subpackages__"])

cc_library(
    name = "macros",
    hdrs = ["macros.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        "@eigen_archive//:eigen3",
        "@tsl//tsl/platform:ml_dtypes",
        "@tsl//tsl/platform:tstring",
        "@xla//xla/python/ifrt",
    ],
)

cc_library(
    name = "op_common",
    hdrs = ["op_common.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        ":macros",
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@tsl//tsl/platform:tstring",
        "@xla//xla/python/ifrt",
        "@xla_serving//data/kernels/util:bcast",
        "@xla_serving//data/processor/cc/api:registry",
        "@xla_serving//data/processor/cc/api:transform_context",
        "@xla_serving//data/processor/cc/core:transform",
        "@xla_serving//data/processor/proto:transform_config_cc_proto",
        "@xla_serving//data/resource_manager/cc:resource",
        "@xla_serving//runtime/core:host_buffer",
    ],
)

alias(
    name = "op_common_test",
    actual = "@xla_serving//data/kernels/math:op_common_test",
)

cc_library(
    name = "test_utils",
    testonly = True,
    hdrs = ["test_utils.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        ":macros",
        "@abseil-cpp//absl/types:span",
        "@xla//xla/python/ifrt",
        "@xla_serving//runtime/core:host_buffer",
    ],
)
