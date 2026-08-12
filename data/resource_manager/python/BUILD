# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    "@neptune//data:neptune_data_internal",
])

pybind_library(
    name = "resource_pybind_lib",
    hdrs = ["resource_pybind.h"],
    deps = [
        "@abseil-cpp//absl/functional:any_invocable",
        "@abseil-cpp//absl/status",
        "@neptune//data/resource_manager/cc:asset",
        "@neptune//data/resource_manager/cc:resource",
        "@pybind11_abseil//pybind11_abseil:status_casters",
    ],
)

pybind_extension(
    name = "resource_manager_pybind",
    srcs = ["resource_manager_pybind.cc"],
    dynamic_deps = ["@neptune//data:neptune_data_core"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../.."],
    deps = [
        ":resource_pybind_lib",
        "@abseil-cpp//absl/strings",
        "@neptune//data/resource_manager/cc:asset",
        "@neptune//data/resource_manager/cc:resource",
        "@neptune//data/resource_manager/cc:resource_manager",
        "@neptune//data/resource_manager/proto:resource_manager_cc_proto",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
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
        "@neptune//data/resource_manager/proto:resource_manager_py_pb2",
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
        "@neptune//data/resource_manager/proto:resource_manager_py_pb2",
        "@neptune//data/resource_manager/python/examples:another_concrete_resource_pybind",
        "@neptune//data/resource_manager/python/examples:concrete_resource_pybind",
        "@neptune//data/test_utils",
    ],
)

py_binary(
    name = "generate_pybind",
    srcs = ["generate_pybind.py"],
)
