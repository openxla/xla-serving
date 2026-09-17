load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    "@xla_serving//data:xla_serving_data_internal",
])

pybind_library(
    name = "resource_pybind_lib",
    hdrs = ["resource_pybind.h"],
    deps = [
        "@abseil-cpp//absl/functional:any_invocable",
        "@abseil-cpp//absl/status",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@xla_serving//data/resource_manager/cc:asset",
        "@xla_serving//data/resource_manager/cc:resource",
    ],
)

pybind_extension(
    name = "resource_manager_pybind",
    srcs = ["resource_manager_pybind.cc"],
    dynamic_deps = ["@xla_serving//data:xla_serving_data_core"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../.."],
    deps = [
        ":resource_pybind_lib",
        "@abseil-cpp//absl/strings",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
        "@xla_serving//data/resource_manager/cc:asset",
        "@xla_serving//data/resource_manager/cc:resource",
        "@xla_serving//data/resource_manager/cc:resource_manager",
        "@xla_serving//data/resource_manager/proto:resource_manager_cc_proto",
    ],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    visibility = ["//visibility:private"],
)

py_library(
    name = "resource_manager",
    srcs = ["resource_manager.py"],
    data = ["resource_manager_pybind.pyi"],
    deps = [
        ":resource_manager_pybind",
        "@xla_serving//data/resource_manager/proto:resource_manager_py_pb2",
    ],
)

py_test(
    name = "resource_manager_test",
    srcs = ["resource_manager_test.py"],
    deps = [
        ":resource_manager",
        "@absl_py//absl/logging",
        "@absl_py//absl/testing:absltest",
        "@com_google_protobuf//:protobuf",
        "@xla_serving//data/resource_manager/proto:resource_manager_py_pb2",
        "@xla_serving//data/resource_manager/python/examples:another_concrete_resource_pybind",
        "@xla_serving//data/resource_manager/python/examples:concrete_resource_pybind",
        "@xla_serving//data/test_utils",
    ],
)

py_binary(
    name = "generate_pybind",
    srcs = ["generate_pybind.py"],
)
