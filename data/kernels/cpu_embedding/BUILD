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
load("@neptune//data/processor/python:build_defs.bzl", "transform_pybind_library")
load("@neptune//data/resource_manager/python:build_defs.bzl", "resource_pybind_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "safe_int_utils",
    hdrs = ["safe_int_utils.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
)

py_library(
    name = "cpu_embedding_module",
    srcs = ["cpu_embedding_module.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":cpu_embedding_lookup_transform_pybind",
        ":cpu_embedding_py_pb2",
        ":cpu_embedding_resource_pybind",
        ":cpu_embedding_serializer_pybind",
        "@com_google_protobuf//:protobuf",
        "@neptune//data/processor/python:composite_processor_pybind",
        "@neptune//data/resource_manager/python:resource_manager",
        "@neptune_pypi//jax",
        "@neptune_pypi//jaxtyping",
        "@neptune_pypi//numpy",
    ],
)

cc_library(
    name = "cpu_embedding_resource",
    srcs = ["cpu_embedding_resource.cc"],
    hdrs = ["cpu_embedding_resource.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        ":cpu_embedding_cc_proto",
        ":safe_int_utils",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@neptune//data/cpu_embedding/cc:array",
        "@neptune//data/cpu_embedding/cc:cpu_embedding_engine",
        "@neptune//data/cpu_embedding/cc:embedding_table",
        "@neptune//data/cpu_embedding/cc:feature_spec",
        "@neptune//data/resource_manager/cc:asset",
        "@neptune//data/resource_manager/cc:resource",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:status_macros",
    ],
    alwayslink = 1,
)

cc_test(
    name = "cpu_embedding_resource_test",
    srcs = ["cpu_embedding_resource_test.cc"],
    defines = [
        "ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS",
        "ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS",
    ],
    deps = [
        ":cpu_embedding_cc_proto",
        ":cpu_embedding_resource",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
        "@neptune//data/cpu_embedding/cc:cpu_embedding_engine",
        "@neptune//data/resource_manager/cc:asset",
        "@neptune//data/resource_manager/cc:resource_manager",
        "@neptune//data/test_utils:status_macros",
        "@tsl//tsl/platform:path",
        "@xla//xla/tsl/platform:env",
    ],
)

proto_library(
    name = "cpu_embedding_proto",
    srcs = ["cpu_embedding.proto"],
)

cc_proto_library(
    name = "cpu_embedding_cc_proto",
    deps = [":cpu_embedding_proto"],
)

py_proto_library(
    name = "cpu_embedding_py_pb2",
    deps = [":cpu_embedding_proto"],
)

cc_library(
    name = "cpu_embedding_lookup_transform",
    srcs = ["cpu_embedding_lookup_transform.cc"],
    hdrs = ["cpu_embedding_lookup_transform.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        ":cpu_embedding_cc_proto",
        ":cpu_embedding_resource",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
        "@neptune//data/processor/cc/api:registry",
        "@neptune//data/processor/cc/api:transform_context",
        "@neptune//data/processor/cc/api:type_info",
        "@neptune//data/processor/cc/core:transform",
        "@neptune//data/processor/proto:transform_config_cc_proto",
        "@neptune//data/resource_manager/cc:resource",
        "@neptune//runtime/core:host_buffer",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/platform:status_macros",
    ],
    alwayslink = 1,
)

cc_test(
    name = "cpu_embedding_lookup_transform_test",
    srcs = ["cpu_embedding_lookup_transform_test.cc"],
    defines = [
        "ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS",
        "ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS",
    ],
    deps = [
        ":cpu_embedding_lookup_transform",
        ":cpu_embedding_resource",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
        "@neptune//data/cpu_embedding/cc:array",
        "@neptune//data/cpu_embedding/cc:cpu_embedding_engine",
        "@neptune//data/cpu_embedding/cc:embedding_table",
        "@neptune//data/cpu_embedding/cc:feature_spec",
        "@neptune//data/processor/cc/api:transform_context",
        "@neptune//data/processor/cc/api:value",
        "@neptune//data/resource_manager/cc:asset",
        "@neptune//data/resource_manager/cc:resource",
        "@neptune//data/test_utils:status_macros",
        "@neptune//runtime/core:host_buffer",
        "@xla//xla/python/ifrt",
    ],
)

resource_pybind_library(
    name = "cpu_embedding_resource_pybind",
    srcs = ["cpu_embedding_resource.cc"],
    hdrs = ["cpu_embedding_resource.h"],
    dynamic_deps = ["@neptune//data:neptune_data_core"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../.."],
    deps = [
        ":cpu_embedding_resource",
    ],
)

transform_pybind_library(
    name = "cpu_embedding_lookup_transform_pybind",
    srcs = ["cpu_embedding_lookup_transform.cc"],
    hdrs = ["cpu_embedding_lookup_transform.h"],
    dynamic_deps = ["@neptune//data:neptune_data_core"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../.."],
    deps = [
        ":cpu_embedding_lookup_transform",
    ],
)

cc_library(
    name = "cpu_embedding_serializer",
    srcs = ["cpu_embedding_serializer.cc"],
    hdrs = ["cpu_embedding_serializer.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        ":cpu_embedding_resource",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@neptune//data/cpu_embedding/cc:array",
        "@neptune//data/resource_manager/cc:asset",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:status_macros",
    ],
)

pybind_extension(
    name = "cpu_embedding_serializer_pybind",
    srcs = ["cpu_embedding_serializer_pybind.cc"],
    dynamic_deps = ["@neptune//data:neptune_data_core"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../.."],
    deps = [
        ":cpu_embedding_resource",
        ":cpu_embedding_serializer",
        "@neptune//data/cpu_embedding/python:array_caster",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:status_casters",
    ],
)
