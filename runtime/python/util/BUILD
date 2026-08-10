load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

py_library(
    name = "model_exporter",
    srcs = ["model_exporter.py"],
    visibility = [
        "@neptune//:__subpackages__",
    ],
    deps = [
        "@neptune_pypi//etils_epath",
        "@neptune_pypi//jax",
        "@neptune_pypi//jaxtyping",
        "@neptune_pypi//orbax_checkpoint",
    ],
)
