load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@rules_python//python:py_test.bzl", "py_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "cpu_embedding",
    srcs = ["cpu_embedding.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@xla_serving//data/kernels/cpu_embedding:cpu_embedding_lookup_transform_pybind",
        "@xla_serving//data/kernels/cpu_embedding:cpu_embedding_module",
        "@xla_serving//data/kernels/cpu_embedding:dense_cpu_embedding_lookup_transform_pybind",
        "@xla_serving//data/processor/python:composite_processor_pybind",
        "@xla_serving//data/processor/python:transform",
        "@xla_serving_pypi//jax",
    ],
)

py_test(
    name = "cpu_embedding_test",
    srcs = ["cpu_embedding_test.py"],
    deps = [
        ":cpu_embedding",
        "@absl_py//absl/testing:absltest",
        "@com_google_protobuf//:protobuf",
        "@xla_serving//data/kernels/cpu_embedding:cpu_embedding_py_pb2",
        "@xla_serving//data/kernels/cpu_embedding:cpu_embedding_serializer_pybind",
        "@xla_serving//data/kernels/cpu_embedding:unpack_host_buffer_vector_module",
        "@xla_serving//data/processor/python:composite_processor_pybind",
        "@xla_serving_pypi//jax",
        "@xla_serving_pypi//numpy",
    ],
)

py_library(
    name = "packing",
    srcs = ["packing.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@xla_serving//data/kernels/cpu_embedding:pack_host_buffer_vector_module",
        "@xla_serving//data/kernels/cpu_embedding:unpack_host_buffer_vector_module",
    ],
)

py_library(
    name = "bucket_pad",
    srcs = ["bucket_pad.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@xla_serving//data/kernels/bucket_pad:bucket_pad_transform_pybind",
        "@xla_serving//data/processor/cc/utils:default_value_py_pb2",
        "@xla_serving//data/processor/python:transform",
    ],
)

py_test(
    name = "bucket_pad_test",
    srcs = ["bucket_pad_test.py"],
    deps = [
        ":bucket_pad",
        "@absl_py//absl/testing:absltest",
        "@xla_serving//data/processor/python:composite_processor_pybind",
        "@xla_serving_pypi//numpy",
    ],
)
