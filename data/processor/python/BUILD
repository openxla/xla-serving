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
    "@neptune//:__subpackages__",
])

pybind_extension(
    name = "composite_processor_pybind",
    srcs = ["composite_processor_pybind.cc"],
    dynamic_deps = ["@neptune//data:neptune_data_core"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../.."],
    deps = [
        ":transform_pybind",
        "@abseil-cpp//absl/base:fast_type_id",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@com_google_protobuf//:any_cc_proto",
        "@dlpack",
        "@neptune//data/processor/cc/api:transform_context",
        "@neptune//data/processor/cc/api:type_info",
        "@neptune//data/processor/cc/api:value",
        "@neptune//data/processor/cc/core:composite_processor",
        "@neptune//data/processor/cc/core:transform",
        "@neptune//data/processor/cc/core/graph:execution_graph",
        "@neptune//data/processor/cc/utils:buffer_utils",
        "@neptune//data/processor/proto:transform_config_cc_proto",
        "@neptune//data/resource_manager/proto:resource_manager_cc_proto",
        "@neptune//runtime/core:host_buffer",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
        "@tsl//tsl/platform:tstring",
        "@xla//xla/python:types",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/platform:status_macros",
    ],
)

pybind_extension(
    name = "composite_processor_test_utils_pybind",
    testonly = True,
    srcs = ["composite_processor_test_utils_pybind.cc"],
    deps = [
        "@neptune//data/processor/cc/core:transform",
        "@neptune//data/processor/cc/utils:buffer_utils",
        "@neptune//runtime/core:host_buffer",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)

py_test(
    name = "composite_processor_test",
    srcs = ["composite_processor_test.py"],
    env = {
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_PLATFORMS": "cpu",
    },
    deps = [
        ":composite_processor_pybind",
        ":composite_processor_test_utils_pybind",
        "@absl_py//absl/testing:absltest",
        "@neptune//data/processor/proto:transform_config_py_pb2",
        "@neptune//data/processor/python/examples:add_from_resource_transform_pybind",
        "@neptune//data/processor/python/examples:add_host_buffer_transform_pybind",
        "@neptune//data/processor/python/examples:add_int_transform_pybind",
        "@neptune//data/processor/python/examples:add_two_transform_pybind",
        "@neptune//data/processor/python/examples:int_resource_pybind",
        "@neptune//data/processor/python/examples:make_dict_transform_pybind",
        "@neptune//data/processor/python/examples:multiply_by_five_transform_pybind",
        "@neptune//data/processor/python/examples:suffix_transform_pybind",
        "@neptune//data/resource_manager/python:resource_manager",
        "@neptune_pypi//jax",
        "@neptune_pypi//ml_dtypes",  # build_cleaner: keep
        "@neptune_pypi//numpy",
    ],
)

pybind_library(
    name = "transform_pybind",
    hdrs = ["transform_pybind.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@neptune//data/processor/cc/api:transform_context",
        "@neptune//data/processor/cc/api:type_info",
        "@neptune//data/processor/cc/core:transform",
        "@neptune//data/processor/proto:transform_config_cc_proto",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
)

py_binary(
    name = "generate_pybind",
    srcs = ["generate_pybind.py"],
)

py_test(
    name = "generate_pybind_test",
    srcs = [
        "generate_pybind.py",
        "generate_pybind_test.py",
    ],
    deps = ["@absl_py//absl/testing:absltest"],
)
