load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = [
    "@xla_serving//:__subpackages__",
])

cc_library(
    name = "serving_core_selector",
    srcs = ["serving_core_selector.cc"],
    hdrs = ["serving_core_selector.h"],
    deps = [
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/synchronization",
        "@xla//xla/tsl/framework:serving_device_selector",
        "@xla//xla/tsl/lib/monitoring:counter",
    ],
)

cc_test(
    name = "serving_core_selector_test",
    srcs = ["serving_core_selector_test.cc"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS"],
    deps = [
        ":serving_core_selector",
        "@abseil-cpp//absl/strings",
        "@googletest//:gtest_main",
        "@xla//xla/tsl/framework:serving_device_selector",
        "@xla//xla/tsl/framework/test_util:mock_serving_device_selector",
        "@xla//xla/tsl/lib/monitoring:cell_reader",
    ],
)
