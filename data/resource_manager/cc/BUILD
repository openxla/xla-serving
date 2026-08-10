load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = [
    "@neptune//data:neptune_data_internal",
])

cc_library(
    name = "asset",
    srcs = ["asset.cc"],
    hdrs = ["asset.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@xla//xla/tsl/platform:env",
    ],
)

cc_test(
    name = "asset_test",
    srcs = ["asset_test.cc"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS"],
    deps = [
        ":asset",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@googletest//:gtest_main",
        "@tsl//tsl/platform:path",
        "@xla//xla/tsl/platform:env",
    ],
)

# TODO: Remove this as it looks like it will no longer be needed.
cc_library(
    name = "asset_manager",
    srcs = ["asset_manager.cc"],
    hdrs = ["asset_manager.h"],
    deps = [
        ":asset",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
    ],
)

# TODO: Remove this as it looks like it will no longer be needed.
cc_test(
    name = "asset_manager_test",
    srcs = ["asset_manager_test.cc"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS"],
    deps = [
        ":asset",
        ":asset_manager",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@googletest//:gtest_main",
        "@neptune//data/test_utils:status_macros",
    ],
)

cc_library(
    name = "resource",
    srcs = ["resource.cc"],
    hdrs = [
        "resource.h",
        "resource_internal.h",
    ],
    deps = [
        ":asset",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/functional:any_invocable",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@neptune//data/resource_manager/proto:resource_manager_cc_proto",
        "@xla//xla/tsl/platform:status_macros",
    ],
)

cc_test(
    name = "resource_test",
    srcs = ["resource_test.cc"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS"],
    deps = [
        ":asset",
        ":resource",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@googletest//:gtest_main",
        "@neptune//data/test_utils:status_macros",
    ],
)

cc_library(
    name = "resource_manager",
    srcs = ["resource_manager.cc"],
    hdrs = ["resource_manager.h"],
    deps = [
        ":asset",
        ":resource",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/types:span",
        "@neptune//data/resource_manager/proto:resource_manager_cc_proto",
        "@xla//xla/tsl/platform:status_macros",
    ],
)

cc_test(
    name = "resource_manager_test",
    srcs = ["resource_manager_test.cc"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS"],
    deps = [
        ":asset",
        ":resource",
        ":resource_manager",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/strings:string_view",
        "@com_google_protobuf//:differencer",
        "@com_google_protobuf//:protobuf",
        "@googletest//:gtest_main",
        "@neptune//data/resource_manager/proto:resource_manager_cc_proto",
        "@neptune//data/test_utils:proto_matchers",
        "@neptune//data/test_utils:status_macros",
    ],
)
