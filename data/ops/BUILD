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
