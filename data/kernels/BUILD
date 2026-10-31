load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@xla_serving//data/processor/python:build_defs.bzl", "transform_pybind_library")

package(default_visibility = ["@xla_serving//:__subpackages__"])

filegroup(
    name = "numerical_transforms_srcs",
    srcs = [
        "numerical_transforms.cc",
        "@xla_serving//data/kernels/bitwise:bitwise_ops_srcs",
        "@xla_serving//data/kernels/math:math_ops_srcs",
    ],
)

cc_library(
    name = "bounds_check",
    hdrs = ["bounds_check.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        "@abseil-cpp//absl/base:core_headers",
        "@eigen_archive//:eigen3",
    ],
)

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
    name = "transpose_functor",
    srcs = ["transpose_functor_cpu.cc"],
    hdrs = ["transpose_functor.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@eigen_archive//:eigen3",
        "@tsl//tsl/platform:tstring",
        "@xla//xla/python/ifrt",
        "@xla_serving//data/kernels/util:ttypes",
        "@xla_serving//runtime/core:host_buffer",
    ],
)

cc_test(
    name = "transpose_util_test",
    srcs = ["transpose_util_test.cc"],
    defines = [
        "ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS",
        "ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS",
    ],
    deps = [
        ":transpose_functor",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
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

cc_library(
    name = "test_utils",
    testonly = True,
    hdrs = ["test_utils.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        ":macros",
        "@abseil-cpp//absl/types:span",
        "@eigen_archive//:eigen3",
        "@tsl//tsl/platform:tstring",
        "@xla//xla/python/ifrt",
        "@xla_serving//runtime/core:host_buffer",
    ],
)

cc_library(
    name = "numerical_transforms",
    srcs = [":numerical_transforms_srcs"],
    hdrs = [
        "numerical_transforms.h",
        "@xla_serving//data/kernels/math:reduction_ops.h",
        "@xla_serving//data/kernels/math:reduction_ops_common.h",
        "@xla_serving//data/kernels/math:scan_ops.h",
    ],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        ":bounds_check",  # build_cleaner: keep
        ":macros",  # build_cleaner: keep
        ":op_common",
        ":transpose_functor",
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/memory",  # build_cleaner: keep
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@eigen_archive//:eigen3",
        "@xla//xla/python/ifrt",
        "@xla_serving//data/kernels/bitwise:op_functors",
        "@xla_serving//data/kernels/math:op_functors",
        "@xla_serving//data/kernels/math:reduction_transform_config_cc_proto",  # build_cleaner: keep
        "@xla_serving//data/kernels/math:scan_transform_config_cc_proto",  # build_cleaner: keep
        "@xla_serving//data/kernels/util:shape_utils",  # build_cleaner: keep
        "@xla_serving//data/kernels/util:ttypes",
        "@xla_serving//data/processor/cc/api:registry",
        "@xla_serving//data/processor/cc/api:shape",
        "@xla_serving//data/processor/cc/api:transform_context",
        "@xla_serving//data/processor/cc/api:type_info",
        "@xla_serving//data/processor/cc/api:value",  # build_cleaner: keep
        "@xla_serving//data/processor/cc/core:transform",
        "@xla_serving//data/processor/proto:transform_config_cc_proto",
        "@xla_serving//data/resource_manager/cc:resource",
        "@xla_serving//runtime/core:host_buffer",
    ],
    alwayslink = 1,
)

transform_pybind_library(
    name = "numerical_transforms_pybind",
    srcs = [":numerical_transforms_srcs"],
    hdrs = [
        "numerical_transforms.h",
        "@xla_serving//data/kernels/math:reduction_ops.h",
        "@xla_serving//data/kernels/math:reduction_ops_common.h",
        "@xla_serving//data/kernels/math:scan_ops.h",
    ],
    dynamic_deps = ["@xla_serving//data:xla_serving_data_core"],
    generate_wrappers = True,
    linkopts = ["-Wl,-rpath,$$ORIGIN/..:$$ORIGIN/../.."],
    deps = [
        ":numerical_transforms",
    ],
)

cc_test(
    name = "numerical_transforms_test",
    srcs = ["numerical_transforms_test.cc"],
    defines = [
        "ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS",
        "ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS",
    ],
    deps = [
        ":numerical_transforms",
        ":test_utils",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_macros",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@eigen_archive//:eigen3",
        "@googletest//:gtest_main",
        "@tsl//tsl/platform:tstring",
        "@xla//xla/python/ifrt",
        "@xla_serving//data/processor/cc/api:shape",
        "@xla_serving//data/processor/cc/api:transform_context",
        "@xla_serving//data/test_utils:shape",
        "@xla_serving//data/test_utils:status_macros",
        "@xla_serving//runtime/core:host_buffer",
    ],
)
