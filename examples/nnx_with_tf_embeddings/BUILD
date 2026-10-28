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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

package(default_visibility = [
    "@xla_serving//examples:__subpackages__",
])

cc_binary(
    name = "client",
    srcs = ["client.cc"],
    deps = [
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/flags:parse",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@grpc//:grpc++",
        "@xla//xla/python/ifrt:dtype_proto_cc",
        "@xla//xla/python/ifrt:shape_proto_cc",
        "@xla//xla/tsl/platform:statusor",
        "@xla_serving//protos/grpc:host_buffer_cc_proto",
        "@xla_serving//protos/grpc:model_cc_proto",
        "@xla_serving//protos/grpc:predict_cc_proto",
        "@xla_serving//protos/grpc:prediction_log_cc_proto",
        "@xla_serving//protos/grpc:prediction_service_cc_grpc_proto",
        "@xla_serving//runtime/core:host_buffer",
        "@xla_serving//runtime/serving/gcp/auth:utils",
    ],
)
