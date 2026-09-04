load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@rules_python//python:py_test.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "cpu_embedding",
    srcs = ["cpu_embedding.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@neptune//data/kernels/cpu_embedding:cpu_embedding_lookup_transform_pybind",
        "@neptune//data/kernels/cpu_embedding:cpu_embedding_module",
        "@neptune//data/kernels/cpu_embedding:dense_cpu_embedding_lookup_transform_pybind",
        "@neptune//data/processor/python:composite_processor_pybind",
        "@neptune//data/processor/python:transform",
        "@neptune_pypi//jax",
    ],
)

py_test(
    name = "cpu_embedding_test",
    srcs = ["cpu_embedding_test.py"],
    deps = [
        ":cpu_embedding",
        "@absl_py//absl/testing:absltest",
        "@neptune//data/kernels/cpu_embedding:cpu_embedding_serializer_pybind",
        "@neptune//data/kernels/cpu_embedding:unpack_host_buffer_vector_module",
        "@neptune//data/processor/python:composite_processor_pybind",
        "@neptune_pypi//jax",
        "@neptune_pypi//numpy",
    ],
)

py_library(
    name = "packing",
    srcs = ["packing.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@neptune//data/kernels/cpu_embedding:pack_host_buffer_vector_module",
        "@neptune//data/kernels/cpu_embedding:unpack_host_buffer_vector_module",
    ],
)

py_library(
    name = "bucket_pad",
    srcs = ["bucket_pad.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@neptune//data/kernels/bucket_pad:bucket_pad_transform_pybind",
        "@neptune//data/processor/cc/utils:default_value_py_pb2",
        "@neptune//data/processor/python:transform",
    ],
)

py_test(
    name = "bucket_pad_test",
    srcs = ["bucket_pad_test.py"],
    deps = [
        ":bucket_pad",
        "@absl_py//absl/testing:absltest",
        "@neptune//data/processor/python:composite_processor_pybind",
        "@neptune_pypi//numpy",
    ],
)
