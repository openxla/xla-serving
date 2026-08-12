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
load("@rules_proto//proto:defs.bzl", "proto_library")

package(default_visibility = [
    "@neptune//data:neptune_data_internal",
])

proto_library(
    name = "resource_manager_proto",
    srcs = ["resource_manager.proto"],
    option_deps = ["@com_google_protobuf//:cpp_features_proto"],
)

py_proto_library(
    name = "resource_manager_py_pb2",
    deps = [":resource_manager_proto"],
)

cc_proto_library(
    name = "resource_manager_cc_proto",
    deps = [":resource_manager_proto"],
)
