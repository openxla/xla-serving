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

load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:py_proto_library.bzl", "py_proto_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(default_visibility = ["@neptune//data:neptune_data_internal"])

cc_library(
    name = "buffer_utils",
    hdrs = ["buffer_utils.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = ["@eigen_archive//:eigen3"],
)

proto_library(
    name = "default_value_proto",
    srcs = ["default_value.proto"],
)

cc_proto_library(
    name = "default_value_cc_proto",
    deps = [":default_value_proto"],
)

py_proto_library(
    name = "default_value_py_pb2",
    deps = [":default_value_proto"],
)

cc_library(
    name = "default_value_utils",
    srcs = ["default_value_utils.cc"],
    hdrs = ["default_value_utils.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        ":default_value_cc_proto",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@com_google_protobuf//:any_cc_proto",
        "@com_google_protobuf//:protobuf",
        "@tsl//tsl/platform:tstring",
    ],
)

cc_test(
    name = "default_value_utils_test",
    srcs = ["default_value_utils_test.cc"],
    defines = [
        "ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS",
        "ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS",
    ],
    deps = [
        ":default_value_cc_proto",
        ":default_value_utils",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@com_google_protobuf//:any_cc_proto",
        "@googletest//:gtest_main",
        "@neptune//data/test_utils:status_macros",
    ],
)
