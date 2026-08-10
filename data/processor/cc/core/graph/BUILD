load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["@neptune//data:processor_internal"])

cc_library(
    name = "execution_graph",
    hdrs = ["execution_graph.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@neptune//data/processor/cc/api:value",
        "@neptune//data/processor/cc/core:transform",
        "@neptune//data/processor/proto:transform_config_cc_proto",
        "@xla//xla/tsl/platform:status_macros",
    ],
)

cc_test(
    name = "execution_graph_test",
    srcs = ["execution_graph_test.cc"],
    defines = [
        "ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS",
        "ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS",
    ],
    deps = [
        ":execution_graph",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
        "@neptune//data/processor/cc/api:transform_context",
        "@neptune//data/processor/cc/api:type_info",
        "@neptune//data/processor/cc/core:transform",
        "@neptune//data/processor/proto:transform_config_cc_proto",
        "@neptune//data/test_utils:status_macros",
    ],
)

cc_library(
    name = "graph_validator",
    srcs = ["graph_validator.cc"],
    hdrs = ["graph_validator.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        ":execution_graph",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@xla//xla/tsl/platform:status_macros",
    ],
)

cc_test(
    name = "graph_validator_test",
    srcs = ["graph_validator_test.cc"],
    defines = [
        "ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS",
        "ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS",
    ],
    deps = [
        ":execution_graph",
        ":graph_validator",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@googletest//:gtest_main",
        "@neptune//data/processor/cc/api:registry",
        "@neptune//data/processor/cc/core:transform",
        "@neptune//data/processor/proto:transform_config_cc_proto",
        "@neptune//data/test_utils:status_macros",
    ],
)

cc_library(
    name = "graph_executor",
    srcs = ["graph_executor.cc"],
    hdrs = ["graph_executor.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        ":execution_graph",
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:vlog_is_on",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@neptune//data/processor/cc/api:transform_context",
        "@neptune//data/processor/cc/api:value",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)

cc_test(
    name = "graph_executor_test",
    srcs = ["graph_executor_test.cc"],
    defines = [
        "ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS",
        "ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS",
    ],
    deps = [
        ":execution_graph",
        ":graph_executor",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
        "@neptune//data/processor/cc/api:registry",
        "@neptune//data/processor/cc/api:transform_context",
        "@neptune//data/processor/cc/api:type_info",
        "@neptune//data/processor/cc/api:value",
        "@neptune//data/processor/cc/core:transform",
        "@neptune//data/processor/proto:transform_config_cc_proto",
        "@neptune//data/test_utils:status_macros",
    ],
)
