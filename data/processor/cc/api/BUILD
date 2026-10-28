load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["@xla_serving//data:processor_internal"])

cc_library(
    name = "type_info",
    hdrs = ["type_info.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        "@abseil-cpp//absl/base:fast_type_id",
        "@abseil-cpp//absl/strings",
    ],
)

cc_library(
    name = "value",
    srcs = ["value.cc"],
    hdrs = ["value.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        ":type_info",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/strings",
    ],
)

cc_library(
    name = "shape",
    srcs = ["shape.cc"],
    hdrs = ["shape.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@xla//xla/python/ifrt",
        "@xla_serving//data/processor/proto:transform_config_cc_proto",
    ],
)

cc_library(
    name = "transform_context",
    hdrs = ["transform_context.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        ":value",
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@xla//xla/python/ifrt",
    ],
)

cc_library(
    name = "type_converter",
    hdrs = ["type_converter.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        ":type_info",
        ":value",
        "@abseil-cpp//absl/status:statusor",
    ],
)

cc_library(
    name = "registry",
    hdrs = ["registry.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        "@xla_serving//data/processor/cc/core:transform",
    ],
)

cc_test(
    name = "value_test",
    srcs = ["value_test.cc"],
    defines = [
        "ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS",
        "ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS",
    ],
    deps = [
        ":type_info",
        ":value",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "shape_test",
    srcs = ["shape_test.cc"],
    defines = [
        "ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS",
        "ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS",
    ],
    deps = [
        ":shape",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/hash:hash_testing",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
        "@xla//xla/python/ifrt",
        "@xla_serving//data/processor/proto:transform_config_cc_proto",
        "@xla_serving//data/test_utils:shape",
        "@xla_serving//data/test_utils:status_macros",
    ],
)

cc_test(
    name = "type_converter_test",
    srcs = ["type_converter_test.cc"],
    defines = [
        "ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS",
        "ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS",
    ],
    deps = [
        ":type_converter",
        ":value",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@googletest//:gtest_main",
        "@xla_serving//data/test_utils:status_macros",
    ],
)
