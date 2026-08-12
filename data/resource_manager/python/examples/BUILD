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

load("@neptune//data/resource_manager/python:build_defs.bzl", "resource_pybind_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = [
    "@neptune//:__subpackages__",
    "@neptune//data:neptune_data_internal",
])

cc_library(
    name = "concrete_resource",
    hdrs = ["concrete_resource.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        "@abseil-cpp//absl/status",
        "@neptune//data/resource_manager/cc:resource",
    ],
    alwayslink = 1,
)

resource_pybind_library(
    name = "concrete_resource_pybind",
    testonly = 1,
    hdrs = ["concrete_resource.h"],
    dynamic_deps = ["@neptune//data:neptune_data_test_core"],
    deps = [":concrete_resource"],
)

cc_library(
    name = "another_concrete_resource",
    srcs = ["another_concrete_resource.cc"],
    hdrs = ["another_concrete_resource.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        "@abseil-cpp//absl/status",
        "@neptune//data/resource_manager/cc:resource",
    ],
    alwayslink = 1,
)

resource_pybind_library(
    name = "another_concrete_resource_pybind",
    testonly = 1,
    srcs = ["another_concrete_resource.cc"],
    hdrs = ["another_concrete_resource.h"],
    dynamic_deps = ["@neptune//data:neptune_data_test_core"],
    deps = [":another_concrete_resource"],
)
