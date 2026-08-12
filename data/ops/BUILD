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

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@rules_python//python:py_test.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "cpu_embedding",
    srcs = ["cpu_embedding.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@neptune//data/kernels/cpu_embedding:cpu_embedding_module",
    ],
)

py_test(
    name = "cpu_embedding_test",
    srcs = ["cpu_embedding_test.py"],
    deps = [
        ":cpu_embedding",
        "@absl_py//absl/testing:absltest",
        "@neptune//data/kernels/cpu_embedding:cpu_embedding_serializer_pybind",
        "@neptune//data/processor/python:composite_processor_pybind",
        "@neptune_pypi//jax",
        "@neptune_pypi//numpy",
    ],
)
