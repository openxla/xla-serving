load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["@xla_serving//:__subpackages__"])

exports_files(glob(["*.h"]))

cc_library(
    name = "op_functors",
    hdrs = [
        "op_functors.h",
        "special_math_impl.h",
    ],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@eigen_archive//:eigen3",
        "@tsl//tsl/platform:tstring",
    ],
)

cc_test(
    name = "op_common_test",
    srcs = ["op_common_test.cc"],
    defines = [
        "ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS",
        "ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS",
    ],
    deps = [
        ":op_functors",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@googletest//:gtest_main",
        "@tsl//tsl/platform:tstring",
        "@xla//xla/python/ifrt",
        "@xla_serving//data/kernels:op_common",
        "@xla_serving//data/kernels:test_utils",
        "@xla_serving//data/kernels/bitwise:op_functors",
        "@xla_serving//data/processor/cc/api:transform_context",
        "@xla_serving//data/test_utils:status_macros",
        "@xla_serving//runtime/core:host_buffer",
    ],
)
