load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_python//python:py_library.bzl", "py_library")

package(default_visibility = ["@neptune//data:neptune_data_internal"])

cc_library(
    name = "proto_matchers",
    testonly = True,
    hdrs = ["proto_matchers.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        "@abseil-cpp//absl/strings:string_view",
        "@com_google_protobuf//:differencer",
        "@com_google_protobuf//:protobuf",
        "@googletest//:gtest",
    ],
)

cc_library(
    name = "status_macros",
    testonly = True,
    hdrs = ["status_macros.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        "@googletest//:gtest",
    ],
)

py_library(
    name = "test_utils",
    testonly = True,
    srcs = ["test_utils.py"],
    deps = [
        "@com_google_protobuf//:protobuf",
    ],
)
