load("@neptune//data/processor/python:build_defs.bzl", "transform_pybind_library")
load("@neptune//data/resource_manager/python:build_defs.bzl", "resource_pybind_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "int_resource",
    srcs = ["int_resource.cc"],
    hdrs = ["int_resource.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@neptune//data/resource_manager/cc:resource",
    ],
    alwayslink = True,
)

resource_pybind_library(
    name = "int_resource_pybind",
    testonly = 1,
    srcs = ["int_resource.cc"],
    hdrs = ["int_resource.h"],
    dynamic_deps = ["@neptune//data:neptune_data_test_core"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../../../neptune_data_test:$$ORIGIN/../../../neptune_data"],
    deps = [":int_resource"],
)

cc_library(
    name = "add_int_from_resource_transform",
    srcs = ["add_int_from_resource_transform.cc"],
    hdrs = ["add_int_from_resource_transform.h"],
    deps = [
        ":int_resource",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
        "@neptune//data/processor/cc/api:registry",
        "@neptune//data/processor/cc/core:transform",
        "@neptune//data/processor/proto:transform_config_cc_proto",
        "@neptune//data/resource_manager/cc:resource",
        "@neptune//data/resource_manager/cc:resource_manager",
        "@neptune//data/resource_manager/proto:resource_manager_cc_proto",
        "@neptune//runtime/core:host_buffer",
        "@xla//xla/python/ifrt",
    ],
    alwayslink = True,
)

transform_pybind_library(
    name = "add_int_from_resource_transform_pybind",
    testonly = 1,
    srcs = ["add_int_from_resource_transform.cc"],
    hdrs = ["add_int_from_resource_transform.h"],
    dynamic_deps = ["@neptune//data:neptune_data_test_core"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../../../neptune_data_test:$$ORIGIN/../../../neptune_data"],
    deps = [":add_int_from_resource_transform"],
)

cc_library(
    name = "add_two_input_buffers_transform",
    srcs = ["add_two_input_buffers_transform.cc"],
    hdrs = ["add_two_input_buffers_transform.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@neptune//data/processor/cc/api:registry",
        "@neptune//data/processor/cc/core:transform",
        "@neptune//data/processor/proto:transform_config_cc_proto",
        "@neptune//runtime/core:host_buffer",
        "@xla//xla/python/ifrt",
    ],
    alwayslink = True,
)

transform_pybind_library(
    name = "add_two_input_buffers_transform_pybind",
    testonly = 1,
    srcs = ["add_two_input_buffers_transform.cc"],
    hdrs = ["add_two_input_buffers_transform.h"],
    dynamic_deps = ["@neptune//data:neptune_data_test_core"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../../../neptune_data_test:$$ORIGIN/../../../neptune_data"],
    deps = [":add_two_input_buffers_transform"],
)

cc_library(
    name = "example_transforms_and_resources",
    visibility = ["@neptune//:__subpackages__"],
    deps = [
        ":add_int_from_resource_transform",
        ":add_two_input_buffers_transform",
        ":int_resource",
    ],
    alwayslink = True,
)

cc_library(
    name = "composite_processor_example_support",
    visibility = ["@neptune//:__subpackages__"],
    deps = [":example_transforms_and_resources"],
    alwayslink = True,
)

py_library(
    name = "export",
    testonly = 1,
    srcs = ["export.py"],
    deps = [
        ":add_int_from_resource_transform_pybind",
        ":add_two_input_buffers_transform_pybind",
        ":int_resource_pybind",
        "@neptune//data/resource_manager/python:resource_manager",
        "@neptune//neptune_model/_src/core:typing",
        "@neptune//neptune_model/_src/export:jax_model",
        "@neptune//neptune_model/_src/export:main_lib",
        "@neptune//neptune_model/_src/export:serving_config",
        "@neptune//protos:orchestration_py_pb2",
        "@neptune_pypi//jax",
    ],
)

py_test(
    name = "export_test",
    srcs = ["export_test.py"],
    deps = [
        ":export",
        "@absl_py//absl/logging",
        "@absl_py//absl/testing:absltest",
        "@neptune//runtime/python:model_runner",  # buildcleaner: keep
        "@neptune_pypi//numpy",
    ],
)

py_binary(
    name = "run_export",
    srcs = ["run_export.py"],
    deps = [
        ":export",
        "//third_party/py/absl:app",
        "@absl_py//absl/flags",
    ],
)

cc_test(
    name = "add_int_from_resource_transform_test",
    srcs = ["add_int_from_resource_transform_test.cc"],
    deps = [
        ":add_int_from_resource_transform",
        ":int_resource",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@googletest//:gtest_main",
        "@neptune//data/resource_manager/cc:asset",
        "@neptune//runtime/core:host_buffer",
        "@xla//xla/python/ifrt",
    ],
)
