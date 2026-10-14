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
load("@rules_cc//cc:cc_shared_library.bzl", "cc_shared_library")
load("//:version.bzl", "VERSION")
load("@rules_python//python:defs.bzl", "py_library")
load("@rules_python//python:packaging.bzl", "py_wheel")

package(default_visibility = [":default_package_visibility"])

package_group(
    name = "default_package_visibility",
    includes = [
    ],
    packages = [
    ],
)

package_group(
    name = "processor_internal",
    packages = [
        "//...",
    ],
)

package_group(
    name = "xla_serving_data_internal",
    packages = [
        "//...",
    ],
)

cc_shared_library(
    name = "xla_serving_data_core",
    exports_filter = [
        "//:__subpackages__",
        "@abseil-cpp//:__subpackages__",
        "@com_google_protobuf//:__subpackages__",
        "@xla//:__subpackages__",
        "@tsl//:__subpackages__",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//data/kernels:numerical_transforms",
        "//data/kernels/cpu_embedding:cpu_embedding_cc_proto",
        "//data/kernels/cpu_embedding:cpu_embedding_lookup_transform",
        "//data/kernels/cpu_embedding:cpu_embedding_resource",
        "//data/kernels/cpu_embedding:cpu_embedding_serializer",
        "//data/kernels/cpu_embedding:dense_cpu_embedding_lookup_cc_proto",
        "//data/kernels/cpu_embedding:dense_cpu_embedding_lookup_transform",
        "//data/kernels/cpu_embedding:pack_host_buffer_vector_transform",
        "//data/kernels/cpu_embedding:unpack_host_buffer_vector_transform",
        "//data/processor/cc/core:composite_processor",
        "//data/processor/cc/core:transform",
        "//data/processor/cc/core/graph:execution_graph",
        "//data/processor/cc/core/graph:graph_executor",
        "//data/processor/cc/core/graph:graph_validator",
        "//data/resource_manager/cc:asset",
        "//data/resource_manager/cc:resource",
        "//data/resource_manager/cc:resource_manager",
        "//runtime/core:host_buffer",
        "@abseil-cpp//absl/base:fast_type_id",
        "@abseil-cpp//absl/synchronization:synchronization",
        "@com_google_protobuf//:wrappers_cc_proto",
        "@xla//xla:literal",
    ],
)

cc_shared_library(
    name = "xla_serving_data_test_core",
    testonly = 1,
    dynamic_deps = [":xla_serving_data_core"],
    exports_filter = [
        "//:__subpackages__",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//data/processor/python/examples:add_from_resource_transform",
        "//data/processor/python/examples:add_host_buffer_transform",
        "//data/processor/python/examples:add_int_transform",
        "//data/processor/python/examples:add_two_transform",
        "//data/processor/python/examples:dummy_resource_transform",
        "//data/processor/python/examples:int_resource",
        "//data/processor/python/examples:make_dict_transform",
        "//data/processor/python/examples:multiply_by_five_transform",
        "//data/processor/python/examples:suffix_transform",
        "//data/resource_manager/python/examples:another_concrete_resource",
        "//data/resource_manager/python/examples:concrete_resource",
        "//examples/composite_processor:add_int_from_resource_transform",
        "//examples/composite_processor:add_two_input_buffers_transform",
        "//examples/composite_processor:int_resource",
    ],
)

genrule(
    name = "stage_xla_serving_data_pkg",
    srcs = [
        "__init__.py",
        ":xla_serving_data_core",
        "@pybind11_abseil//pybind11_abseil:status.so",
        "//data/ops:bitwise.py",
        "//data/ops:bucket_pad.py",
        "//data/ops:cpu_embedding.py",
        "//data/ops:math.py",
        "//data/ops:packing.py",
        "//data/ops:reduction_ops.py",
        "//data/ops:scan_ops.py",
        "//data/kernels:numerical_transforms_module.py",
        "//data/kernels:numerical_transforms_pybind.so",
        "//data/kernels/bucket_pad:bucket_pad_transform_pybind.so",
        "//data/kernels/bucket_pad:bucket_pad_transform_py_pb2",
        "//data/kernels/cpu_embedding:cpu_embedding_module.py",
        "//data/kernels/cpu_embedding:cpu_embedding_lookup_transform_pybind.so",
        "//data/kernels/cpu_embedding:cpu_embedding_resource_pybind.so",
        "//data/kernels/cpu_embedding:cpu_embedding_serializer_pybind.so",
        "//data/kernels/cpu_embedding:cpu_embedding_py_pb2",
        "//data/kernels/cpu_embedding:dense_cpu_embedding_lookup_transform_pybind.so",
        "//data/kernels/cpu_embedding:dense_cpu_embedding_lookup_py_pb2",
        "//data/kernels/cpu_embedding:pack_host_buffer_vector_module.py",
        "//data/kernels/cpu_embedding:pack_host_buffer_vector_transform_pybind.so",
        "//data/kernels/cpu_embedding:unpack_host_buffer_vector_module.py",
        "//data/kernels/cpu_embedding:unpack_host_buffer_vector_transform_pybind.so",
        "//data/processor/python:composite_processor_pybind.so",
        "//data/processor/python:transform.py",
        "//data/processor/proto:transform_config_py_pb2",
        "//data/processor/cc/utils:default_value_py_pb2",
        "//data/resource_manager/python:resource_manager.py",
        "//data/resource_manager/python:resource_manager_pybind.so",
        "//data/resource_manager/proto:resource_manager_py_pb2",
        "//data/cpu_embedding/python:embedding_utils.py",
    ],
    outs = [
        "xla_serving_data/__init__.py",
        "xla_serving_data/libxla_serving_data_core.so",
        "xla_serving_data/cpu_embedding/__init__.py",
        "xla_serving_data/cpu_embedding/python/__init__.py",
        "xla_serving_data/cpu_embedding/python/embedding_utils.py",
        "xla_serving_data/ops/__init__.py",
        "xla_serving_data/ops/bitwise.py",
        "xla_serving_data/ops/bucket_pad.py",
        "xla_serving_data/ops/cpu_embedding.py",
        "xla_serving_data/ops/math.py",
        "xla_serving_data/ops/packing.py",
        "xla_serving_data/ops/reduction_ops.py",
        "xla_serving_data/ops/scan_ops.py",
        "xla_serving_data/kernels/__init__.py",
        "xla_serving_data/kernels/numerical_transforms_module.py",
        "xla_serving_data/kernels/numerical_transforms_pybind.so",
        "xla_serving_data/kernels/bucket_pad/__init__.py",
        "xla_serving_data/kernels/bucket_pad/bucket_pad_transform_pybind.so",
        "xla_serving_data/kernels/bucket_pad/bucket_pad_transform_pb2.py",
        "xla_serving_data/kernels/cpu_embedding/__init__.py",
        "xla_serving_data/kernels/cpu_embedding/cpu_embedding_module.py",
        "xla_serving_data/kernels/cpu_embedding/cpu_embedding_lookup_transform_pybind.so",
        "xla_serving_data/kernels/cpu_embedding/cpu_embedding_resource_pybind.so",
        "xla_serving_data/kernels/cpu_embedding/cpu_embedding_serializer_pybind.so",
        "xla_serving_data/kernels/cpu_embedding/cpu_embedding_pb2.py",
        "xla_serving_data/kernels/cpu_embedding/dense_cpu_embedding_lookup_transform_pybind.so",
        "xla_serving_data/kernels/cpu_embedding/dense_cpu_embedding_lookup_pb2.py",
        "xla_serving_data/kernels/cpu_embedding/pack_host_buffer_vector_module.py",
        "xla_serving_data/kernels/cpu_embedding/pack_host_buffer_vector_transform_pybind.so",
        "xla_serving_data/kernels/cpu_embedding/unpack_host_buffer_vector_module.py",
        "xla_serving_data/kernels/cpu_embedding/unpack_host_buffer_vector_transform_pybind.so",
        "xla_serving_data/processor/__init__.py",
        "xla_serving_data/processor/python/__init__.py",
        "xla_serving_data/processor/python/composite_processor_pybind.so",
        "xla_serving_data/processor/python/transform.py",
        "xla_serving_data/processor/proto/__init__.py",
        "xla_serving_data/processor/proto/transform_config_pb2.py",
        "xla_serving_data/processor/cc/__init__.py",
        "xla_serving_data/processor/cc/utils/__init__.py",
        "xla_serving_data/processor/cc/utils/default_value_pb2.py",
        "xla_serving_data/resource_manager/__init__.py",
        "xla_serving_data/resource_manager/python/__init__.py",
        "xla_serving_data/resource_manager/python/resource_manager.py",
        "xla_serving_data/resource_manager/python/resource_manager_pybind.so",
        "xla_serving_data/resource_manager/proto/__init__.py",
        "xla_serving_data/resource_manager/proto/resource_manager_pb2.py",
        "xla_serving_data/pybind11_abseil/__init__.py",
        "xla_serving_data/pybind11_abseil/status.so",
    ],
    cmd = """
  mkdir -p $(@D)/xla_serving_data/ops
  mkdir -p $(@D)/xla_serving_data/kernels/bucket_pad
  mkdir -p $(@D)/xla_serving_data/kernels/cpu_embedding
  mkdir -p $(@D)/xla_serving_data/processor/python
  mkdir -p $(@D)/xla_serving_data/processor/proto
  mkdir -p $(@D)/xla_serving_data/processor/cc/utils
  mkdir -p $(@D)/xla_serving_data/resource_manager/python
  mkdir -p $(@D)/xla_serving_data/resource_manager/proto
  mkdir -p $(@D)/xla_serving_data/pybind11_abseil

  cp $(location __init__.py) $(@D)/xla_serving_data/__init__.py
  touch $(@D)/xla_serving_data/ops/__init__.py
  touch $(@D)/xla_serving_data/kernels/__init__.py
  touch $(@D)/xla_serving_data/kernels/bucket_pad/__init__.py
  touch $(@D)/xla_serving_data/kernels/cpu_embedding/__init__.py
  touch $(@D)/xla_serving_data/processor/__init__.py
  touch $(@D)/xla_serving_data/processor/python/__init__.py
  touch $(@D)/xla_serving_data/processor/proto/__init__.py
  touch $(@D)/xla_serving_data/processor/cc/__init__.py
  touch $(@D)/xla_serving_data/processor/cc/utils/__init__.py
  touch $(@D)/xla_serving_data/resource_manager/__init__.py
  touch $(@D)/xla_serving_data/resource_manager/python/__init__.py
  touch $(@D)/xla_serving_data/resource_manager/proto/__init__.py
  touch $(@D)/xla_serving_data/pybind11_abseil/__init__.py

  cp $(location :xla_serving_data_core) $(@D)/xla_serving_data/libxla_serving_data_core.so
  cp $(location //data/ops:bitwise.py) $(@D)/xla_serving_data/ops/bitwise.py
  sed -i 's/from data\\./from xla_serving_data\\./g' $(@D)/xla_serving_data/ops/bitwise.py
  cp $(location //data/ops:bucket_pad.py) $(@D)/xla_serving_data/ops/bucket_pad.py
  sed -i 's/from data\\./from xla_serving_data\\./g' $(@D)/xla_serving_data/ops/bucket_pad.py
  cp $(location //data/ops:cpu_embedding.py) $(@D)/xla_serving_data/ops/cpu_embedding.py
  sed -i 's/from data\\./from xla_serving_data\\./g' $(@D)/xla_serving_data/ops/cpu_embedding.py
  cp $(location //data/ops:math.py) $(@D)/xla_serving_data/ops/math.py
  sed -i 's/from data\\./from xla_serving_data\\./g' $(@D)/xla_serving_data/ops/math.py
  cp $(location //data/ops:packing.py) $(@D)/xla_serving_data/ops/packing.py
  sed -i 's/from data\\./from xla_serving_data\\./g' $(@D)/xla_serving_data/ops/packing.py
  cp $(location //data/ops:reduction_ops.py) $(@D)/xla_serving_data/ops/reduction_ops.py
  sed -i 's/from data\\./from xla_serving_data\\./g' $(@D)/xla_serving_data/ops/reduction_ops.py
  cp $(location //data/ops:scan_ops.py) $(@D)/xla_serving_data/ops/scan_ops.py
  sed -i 's/from data\\./from xla_serving_data\\./g' $(@D)/xla_serving_data/ops/scan_ops.py
  cp $(location //data/kernels:numerical_transforms_module.py) $(@D)/xla_serving_data/kernels/numerical_transforms_module.py
  sed -i 's/from data\\./from xla_serving_data\\./g' $(@D)/xla_serving_data/kernels/numerical_transforms_module.py
  cp $(location //data/kernels:numerical_transforms_pybind.so) $(@D)/xla_serving_data/kernels/numerical_transforms_pybind.so
  cp $(location //data/kernels/bucket_pad:bucket_pad_transform_pybind.so) $(@D)/xla_serving_data/kernels/bucket_pad/bucket_pad_transform_pybind.so
  for f in $(locations //data/kernels/bucket_pad:bucket_pad_transform_py_pb2); do
    if [[ $$f == *bucket_pad_transform_pb2.py ]]; then
      cp $$f $(@D)/xla_serving_data/kernels/bucket_pad/bucket_pad_transform_pb2.py
      sed -i 's/from data\\./from xla_serving_data\\./g' $(@D)/xla_serving_data/kernels/bucket_pad/bucket_pad_transform_pb2.py
      sed -i 's/data\\.kernels\\.bucket_pad/xla_serving_data\\.kernels\\.bucket_pad/g' $(@D)/xla_serving_data/kernels/bucket_pad/bucket_pad_transform_pb2.py
    fi
  done
  cp $(location //data/kernels/cpu_embedding:cpu_embedding_module.py) $(@D)/xla_serving_data/kernels/cpu_embedding/cpu_embedding_module.py
  sed -i 's/from data\\./from xla_serving_data\\./g' $(@D)/xla_serving_data/kernels/cpu_embedding/cpu_embedding_module.py
  cp $(location //data/kernels/cpu_embedding:cpu_embedding_lookup_transform_pybind.so) $(@D)/xla_serving_data/kernels/cpu_embedding/cpu_embedding_lookup_transform_pybind.so
  cp $(location //data/kernels/cpu_embedding:cpu_embedding_resource_pybind.so) $(@D)/xla_serving_data/kernels/cpu_embedding/cpu_embedding_resource_pybind.so
  cp $(location //data/kernels/cpu_embedding:cpu_embedding_serializer_pybind.so) $(@D)/xla_serving_data/kernels/cpu_embedding/cpu_embedding_serializer_pybind.so
  for f in $(locations //data/kernels/cpu_embedding:cpu_embedding_py_pb2); do
    if [[ $$f == *cpu_embedding_pb2.py ]]; then
      cp $$f $(@D)/xla_serving_data/kernels/cpu_embedding/cpu_embedding_pb2.py
      sed -i 's/from data\\./from xla_serving_data\\./g' $(@D)/xla_serving_data/kernels/cpu_embedding/cpu_embedding_pb2.py
      sed -i 's/data\\.kernels\\.cpu_embedding/xla_serving_data\\.kernels\\.cpu_embedding/g' $(@D)/xla_serving_data/kernels/cpu_embedding/cpu_embedding_pb2.py
    fi
  done
  cp $(location //data/kernels/cpu_embedding:dense_cpu_embedding_lookup_transform_pybind.so) $(@D)/xla_serving_data/kernels/cpu_embedding/dense_cpu_embedding_lookup_transform_pybind.so
  for f in $(locations //data/kernels/cpu_embedding:dense_cpu_embedding_lookup_py_pb2); do
    if [[ $$f == *dense_cpu_embedding_lookup_pb2.py ]]; then
      cp $$f $(@D)/xla_serving_data/kernels/cpu_embedding/dense_cpu_embedding_lookup_pb2.py
      sed -i 's/from data\\./from xla_serving_data\\./g' $(@D)/xla_serving_data/kernels/cpu_embedding/dense_cpu_embedding_lookup_pb2.py
      sed -i 's/data\\.kernels\\.cpu_embedding/xla_serving_data\\.kernels\\.cpu_embedding/g' $(@D)/xla_serving_data/kernels/cpu_embedding/dense_cpu_embedding_lookup_pb2.py
    fi
  done
  cp $(location //data/kernels/cpu_embedding:pack_host_buffer_vector_module.py) $(@D)/xla_serving_data/kernels/cpu_embedding/pack_host_buffer_vector_module.py
  sed -i 's/from data\\./from xla_serving_data\\./g' $(@D)/xla_serving_data/kernels/cpu_embedding/pack_host_buffer_vector_module.py
  cp $(location //data/kernels/cpu_embedding:pack_host_buffer_vector_transform_pybind.so) $(@D)/xla_serving_data/kernels/cpu_embedding/pack_host_buffer_vector_transform_pybind.so
  cp $(location //data/kernels/cpu_embedding:unpack_host_buffer_vector_module.py) $(@D)/xla_serving_data/kernels/cpu_embedding/unpack_host_buffer_vector_module.py
  sed -i 's/from data\\./from xla_serving_data\\./g' $(@D)/xla_serving_data/kernels/cpu_embedding/unpack_host_buffer_vector_module.py
  cp $(location //data/kernels/cpu_embedding:unpack_host_buffer_vector_transform_pybind.so) $(@D)/xla_serving_data/kernels/cpu_embedding/unpack_host_buffer_vector_transform_pybind.so
  cp $(location //data/processor/python:composite_processor_pybind.so) $(@D)/xla_serving_data/processor/python/composite_processor_pybind.so
  cp $(location //data/processor/python:transform.py) $(@D)/xla_serving_data/processor/python/transform.py
  sed -i 's/from data\\./from xla_serving_data\\./g' $(@D)/xla_serving_data/processor/python/transform.py
  for f in $(locations //data/processor/proto:transform_config_py_pb2); do
    if [[ $$f == *transform_config_pb2.py ]]; then
      cp $$f $(@D)/xla_serving_data/processor/proto/transform_config_pb2.py
      sed -i 's/from data\\./from xla_serving_data\\./g' $(@D)/xla_serving_data/processor/proto/transform_config_pb2.py
      sed -i 's/data\\.processor\\.proto/xla_serving_data\\.processor\\.proto/g' $(@D)/xla_serving_data/processor/proto/transform_config_pb2.py
    fi
  done
  for f in $(locations //data/processor/cc/utils:default_value_py_pb2); do
    if [[ $$f == *default_value_pb2.py ]]; then
      cp $$f $(@D)/xla_serving_data/processor/cc/utils/default_value_pb2.py
      sed -i 's/from data\\./from xla_serving_data\\./g' $(@D)/xla_serving_data/processor/cc/utils/default_value_pb2.py
      sed -i 's/data\\.processor\\.cc\\.utils/xla_serving_data\\.processor\\.cc\\.utils/g' $(@D)/xla_serving_data/processor/cc/utils/default_value_pb2.py
    fi
  done
  cp $(location //data/resource_manager/python:resource_manager.py) $(@D)/xla_serving_data/resource_manager/python/resource_manager.py
  sed -i 's/from data\\./from xla_serving_data\\./g' $(@D)/xla_serving_data/resource_manager/python/resource_manager.py
  cp $(location //data/resource_manager/python:resource_manager_pybind.so) $(@D)/xla_serving_data/resource_manager/python/resource_manager_pybind.so
  for f in $(locations //data/resource_manager/proto:resource_manager_py_pb2); do
    if [[ $$f == *resource_manager_pb2.py ]]; then
      cp $$f $(@D)/xla_serving_data/resource_manager/proto/resource_manager_pb2.py
      sed -i 's/from data\\./from xla_serving_data\\./g' $(@D)/xla_serving_data/resource_manager/proto/resource_manager_pb2.py
      sed -i 's/data\\.resource_manager\\.proto/xla_serving_data\\.resource_manager\\.proto/g' $(@D)/xla_serving_data/resource_manager/proto/resource_manager_pb2.py
    fi
  done
  cp $(location @pybind11_abseil//pybind11_abseil:status.so) $(@D)/xla_serving_data/pybind11_abseil/status.so
  mkdir -p $(@D)/xla_serving_data/cpu_embedding/python
  touch $(@D)/xla_serving_data/cpu_embedding/__init__.py
  touch $(@D)/xla_serving_data/cpu_embedding/python/__init__.py
  cp $(location //data/cpu_embedding/python:embedding_utils.py) $(@D)/xla_serving_data/cpu_embedding/python/embedding_utils.py
  sed -i 's/from data\\./from xla_serving_data\\./g' $(@D)/xla_serving_data/cpu_embedding/python/embedding_utils.py
  """,
)

py_library(
    name = "xla_serving_data_pkg",
    srcs = [
        "xla_serving_data/__init__.py",
        "xla_serving_data/cpu_embedding/__init__.py",
        "xla_serving_data/cpu_embedding/python/__init__.py",
        "xla_serving_data/cpu_embedding/python/embedding_utils.py",
        "xla_serving_data/kernels/__init__.py",
        "xla_serving_data/kernels/cpu_embedding/__init__.py",
        "xla_serving_data/kernels/cpu_embedding/cpu_embedding_module.py",
        "xla_serving_data/kernels/cpu_embedding/cpu_embedding_pb2.py",
        "xla_serving_data/kernels/cpu_embedding/dense_cpu_embedding_lookup_pb2.py",
        "xla_serving_data/kernels/cpu_embedding/pack_host_buffer_vector_module.py",
        "xla_serving_data/kernels/cpu_embedding/unpack_host_buffer_vector_module.py",
        "xla_serving_data/kernels/numerical_transforms_module.py",
        "xla_serving_data/ops/__init__.py",
        "xla_serving_data/ops/bitwise.py",
        "xla_serving_data/ops/bucket_pad.py",
        "xla_serving_data/ops/cpu_embedding.py",
        "xla_serving_data/ops/math.py",
        "xla_serving_data/ops/packing.py",
        "xla_serving_data/ops/reduction_ops.py",
        "xla_serving_data/ops/scan_ops.py",
        "xla_serving_data/processor/__init__.py",
        "xla_serving_data/processor/cc/__init__.py",
        "xla_serving_data/processor/cc/utils/__init__.py",
        "xla_serving_data/processor/cc/utils/default_value_pb2.py",
        "xla_serving_data/processor/proto/__init__.py",
        "xla_serving_data/processor/proto/transform_config_pb2.py",
        "xla_serving_data/processor/python/__init__.py",
        "xla_serving_data/processor/python/transform.py",
        "xla_serving_data/pybind11_abseil/__init__.py",
        "xla_serving_data/resource_manager/__init__.py",
        "xla_serving_data/resource_manager/proto/__init__.py",
        "xla_serving_data/resource_manager/proto/resource_manager_pb2.py",
        "xla_serving_data/resource_manager/python/__init__.py",
        "xla_serving_data/resource_manager/python/resource_manager.py",
    ],
    data = [
        "xla_serving_data/kernels/cpu_embedding/cpu_embedding_lookup_transform_pybind.so",
        "xla_serving_data/kernels/cpu_embedding/cpu_embedding_pb2.pyi",
        "xla_serving_data/kernels/cpu_embedding/cpu_embedding_resource_pybind.so",
        "xla_serving_data/kernels/cpu_embedding/cpu_embedding_serializer_pybind.so",
        "xla_serving_data/kernels/cpu_embedding/dense_cpu_embedding_lookup_transform_pybind.so",
        "xla_serving_data/kernels/cpu_embedding/pack_host_buffer_vector_transform_pybind.so",
        "xla_serving_data/kernels/cpu_embedding/unpack_host_buffer_vector_transform_pybind.so",
        "xla_serving_data/kernels/numerical_transforms_pybind.so",
        "xla_serving_data/libxla_serving_data_core.so",
        "xla_serving_data/processor/proto/transform_config_pb2.pyi",
        "xla_serving_data/processor/python/composite_processor_pybind.so",
        "xla_serving_data/pybind11_abseil/status.so",
        "xla_serving_data/resource_manager/proto/resource_manager_pb2.pyi",
        "xla_serving_data/resource_manager/python/resource_manager_pybind.so",
    ],
    tags = ["manual"],
    visibility = ["//visibility:public"],
)

py_wheel(
    name = "xla_serving_data",
    abi = select({
        "@rules_python//python/config_settings:is_python_3.11": "cp311",
        "@rules_python//python/config_settings:is_python_3.12": "cp312",
        "@rules_python//python/config_settings:is_python_3.13": "cp313",
        "@rules_python//python/config_settings:is_python_3.14": "cp314",
    }),
    distribution = "xla_serving_data",
    platform = select({
        "//runtime/python:linux_x86_64": "manylinux_2_28_x86_64",
        "//runtime/python:linux_aarch64": "manylinux_2_28_aarch64",
    }),
    python_tag = select({
        "@rules_python//python/config_settings:is_python_3.11": "cp311",
        "@rules_python//python/config_settings:is_python_3.12": "cp312",
        "@rules_python//python/config_settings:is_python_3.13": "cp313",
        "@rules_python//python/config_settings:is_python_3.14": "cp314",
    }),
    strip_path_prefixes = ["data"],
    tags = ["manual"],
    version = VERSION,
    deps = [
        ":stage_xla_serving_data_pkg",
        ":xla_serving_data_pkg",
    ],
)

genrule(
    name = "stage_xla_serving_data_test_pkg",
    testonly = 1,
    srcs = [
        "__init__.py",
        ":xla_serving_data_test_core",
        "//data/processor/python/examples:add_int_transform_pybind.so",
        "//data/processor/python/examples:add_host_buffer_transform_pybind.so",
        "//data/processor/python/examples:add_two_transform_pybind.so",
        "//data/processor/python/examples:multiply_by_five_transform_pybind.so",
        "//data/processor/python/examples:dummy_resource_transform_pybind.so",
        "//data/processor/python/examples:suffix_transform_pybind.so",
        "//data/processor/python/examples:add_from_resource_transform_pybind.so",
        "//data/processor/python/examples:int_resource_pybind.so",
        "//data/processor/python/examples:make_dict_transform_pybind.so",
        "//examples/composite_processor:add_int_from_resource_transform_pybind.so",
        "//examples/composite_processor:add_two_input_buffers_transform_pybind.so",
        "//examples/composite_processor:int_resource_pybind.so",
    ],
    outs = [
        "xla_serving_data_test/__init__.py",
        "xla_serving_data_test/libxla_serving_data_test_core.so",
        "xla_serving_data_test/processor/__init__.py",
        "xla_serving_data_test/processor/python/__init__.py",
        "xla_serving_data_test/processor/python/examples/__init__.py",
        "xla_serving_data_test/processor/python/examples/add_int_transform_pybind.so",
        "xla_serving_data_test/processor/python/examples/add_host_buffer_transform_pybind.so",
        "xla_serving_data_test/processor/python/examples/add_two_transform_pybind.so",
        "xla_serving_data_test/processor/python/examples/multiply_by_five_transform_pybind.so",
        "xla_serving_data_test/processor/python/examples/dummy_resource_transform_pybind.so",
        "xla_serving_data_test/processor/python/examples/suffix_transform_pybind.so",
        "xla_serving_data_test/processor/python/examples/add_from_resource_transform_pybind.so",
        "xla_serving_data_test/processor/python/examples/int_resource_pybind.so",
        "xla_serving_data_test/processor/python/examples/make_dict_transform_pybind.so",
        "xla_serving_data_test/examples/composite_processor/add_int_from_resource_transform_pybind.so",
        "xla_serving_data_test/examples/composite_processor/add_two_input_buffers_transform_pybind.so",
        "xla_serving_data_test/examples/composite_processor/int_resource_pybind.so",
    ],
    cmd = """
  mkdir -p $(@D)/xla_serving_data_test/processor/python/examples
  cp $(location __init__.py) $(@D)/xla_serving_data_test/__init__.py
  touch $(@D)/xla_serving_data_test/processor/__init__.py
  touch $(@D)/xla_serving_data_test/processor/python/__init__.py
  touch $(@D)/xla_serving_data_test/processor/python/examples/__init__.py
  cp $(location :xla_serving_data_test_core) $(@D)/xla_serving_data_test/libxla_serving_data_test_core.so
  cp $(location //data/processor/python/examples:add_int_transform_pybind.so) $(@D)/xla_serving_data_test/processor/python/examples/add_int_transform_pybind.so
  cp $(location //data/processor/python/examples:add_host_buffer_transform_pybind.so) $(@D)/xla_serving_data_test/processor/python/examples/add_host_buffer_transform_pybind.so
  cp $(location //data/processor/python/examples:add_two_transform_pybind.so) $(@D)/xla_serving_data_test/processor/python/examples/add_two_transform_pybind.so
  cp $(location //data/processor/python/examples:multiply_by_five_transform_pybind.so) $(@D)/xla_serving_data_test/processor/python/examples/multiply_by_five_transform_pybind.so
  cp $(location //data/processor/python/examples:dummy_resource_transform_pybind.so) $(@D)/xla_serving_data_test/processor/python/examples/dummy_resource_transform_pybind.so
  cp $(location //data/processor/python/examples:suffix_transform_pybind.so) $(@D)/xla_serving_data_test/processor/python/examples/suffix_transform_pybind.so
  cp $(location //data/processor/python/examples:add_from_resource_transform_pybind.so) $(@D)/xla_serving_data_test/processor/python/examples/add_from_resource_transform_pybind.so
  cp $(location //data/processor/python/examples:int_resource_pybind.so) $(@D)/xla_serving_data_test/processor/python/examples/int_resource_pybind.so
  cp $(location //data/processor/python/examples:make_dict_transform_pybind.so) $(@D)/xla_serving_data_test/processor/python/examples/make_dict_transform_pybind.so
  cp $(location //examples/composite_processor:add_int_from_resource_transform_pybind.so) $(@D)/xla_serving_data_test/examples/composite_processor/add_int_from_resource_transform_pybind.so
  cp $(location //examples/composite_processor:add_two_input_buffers_transform_pybind.so) $(@D)/xla_serving_data_test/examples/composite_processor/add_two_input_buffers_transform_pybind.so
  cp $(location //examples/composite_processor:int_resource_pybind.so) $(@D)/xla_serving_data_test/examples/composite_processor/int_resource_pybind.so
  """,
)

py_library(
    name = "xla_serving_data_test_pkg",
    testonly = 1,
    srcs = [
        "xla_serving_data_test/__init__.py",
        "xla_serving_data_test/processor/__init__.py",
        "xla_serving_data_test/processor/python/__init__.py",
        "xla_serving_data_test/processor/python/examples/__init__.py",
    ],
    data = [
        "xla_serving_data_test/examples/composite_processor/add_int_from_resource_transform_pybind.so",
        "xla_serving_data_test/examples/composite_processor/add_two_input_buffers_transform_pybind.so",
        "xla_serving_data_test/examples/composite_processor/int_resource_pybind.so",
        "xla_serving_data_test/libxla_serving_data_test_core.so",
        "xla_serving_data_test/processor/python/examples/add_from_resource_transform_pybind.so",
        "xla_serving_data_test/processor/python/examples/add_host_buffer_transform_pybind.so",
        "xla_serving_data_test/processor/python/examples/add_int_transform_pybind.so",
        "xla_serving_data_test/processor/python/examples/add_two_transform_pybind.so",
        "xla_serving_data_test/processor/python/examples/dummy_resource_transform_pybind.so",
        "xla_serving_data_test/processor/python/examples/int_resource_pybind.so",
        "xla_serving_data_test/processor/python/examples/make_dict_transform_pybind.so",
        "xla_serving_data_test/processor/python/examples/multiply_by_five_transform_pybind.so",
        "xla_serving_data_test/processor/python/examples/suffix_transform_pybind.so",
    ],
    tags = ["manual"],
    visibility = ["//visibility:public"],
)

py_wheel(
    name = "xla_serving_data_test",
    testonly = 1,
    abi = select({
        "@rules_python//python/config_settings:is_python_3.11": "cp311",
        "@rules_python//python/config_settings:is_python_3.12": "cp312",
        "@rules_python//python/config_settings:is_python_3.13": "cp313",
        "@rules_python//python/config_settings:is_python_3.14": "cp314",
    }),
    distribution = "xla_serving_data_test",
    platform = select({
        "//runtime/python:linux_x86_64": "manylinux_2_28_x86_64",
        "//runtime/python:linux_aarch64": "manylinux_2_28_aarch64",
    }),
    python_tag = select({
        "@rules_python//python/config_settings:is_python_3.11": "cp311",
        "@rules_python//python/config_settings:is_python_3.12": "cp312",
        "@rules_python//python/config_settings:is_python_3.13": "cp313",
        "@rules_python//python/config_settings:is_python_3.14": "cp314",
    }),
    strip_path_prefixes = ["data"],
    tags = ["manual"],
    version = VERSION,
    deps = [
        ":stage_xla_serving_data_test_pkg",
        ":xla_serving_data_test_pkg",
    ],
)
