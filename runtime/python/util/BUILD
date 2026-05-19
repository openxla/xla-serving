load("@rules_python//python:defs.bzl", "py_library", "py_test")

py_library(
    name = "model_exporter",
    srcs = ["model_exporter.py"],
    visibility = [
        "//learning/infra/mira/experimental/orbax_model/interpreter/testdata:__subpackages__",
        "@neptune//:__subpackages__",
    ],
    deps = [
        "//third_party/py/etils/epath",
        "@neptune_pypi//jax",
        "@neptune_pypi//jaxtyping",
        "@neptune_pypi//orbax_checkpoint",
    ],
)
