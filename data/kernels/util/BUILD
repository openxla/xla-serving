load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["@xla_serving//:__subpackages__"])

cc_library(
    name = "bcast",
    hdrs = ["bcast.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        ":overflow",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@eigen_archive//:eigen3",
        "@xla//xla/python/ifrt",
    ],
)

cc_test(
    name = "bcast_test",
    srcs = ["bcast_test.cc"],
    defines = [
        "ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS",
        "ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS",
    ],
    deps = [
        ":bcast",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/strings",
        "@googletest//:gtest_main",
        "@xla_serving//data/test_utils:status_macros",
    ],
)

cc_library(
    name = "overflow",
    hdrs = ["overflow.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        "@abseil-cpp//absl/base:core_headers",
    ],
)

cc_test(
    name = "overflow_test",
    srcs = ["overflow_test.cc"],
    defines = [
        "ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS",
        "ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS",
    ],
    deps = [
        ":overflow",
        "@abseil-cpp//absl/numeric:int128",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "shape_utils",
    srcs = ["shape_utils.cc"],
    hdrs = ["shape_utils.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        ":overflow",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@xla//xla/python/ifrt",
        "@xla_serving//data/kernels:bounds_check",
    ],
)

cc_test(
    name = "shape_utils_test",
    srcs = ["shape_utils_test.cc"],
    defines = [
        "ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS",
        "ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS",
    ],
    deps = [
        ":shape_utils",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
        "@xla//xla/python/ifrt",
    ],
)

cc_library(
    name = "ttypes",
    hdrs = ["ttypes.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        "@abseil-cpp//absl/log:check",
        "@eigen_archive//:eigen3",
        "@xla_serving//runtime/core:host_buffer",
    ],
)

cc_test(
    name = "ttypes_test",
    srcs = ["ttypes_test.cc"],
    defines = [
        "ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS",
        "ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS",
    ],
    deps = [
        ":ttypes",
        "@eigen_archive//:eigen3",
        "@googletest//:gtest_main",
        "@xla_serving//runtime/core:host_buffer",
    ],
)
