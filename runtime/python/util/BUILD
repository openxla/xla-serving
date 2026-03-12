load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

py_library(
    name = "model_exporter",
    srcs = ["model_exporter.py"],
    visibility = [
        "//learning/infra/mira/experimental/orbax_model/interpreter/testdata:__subpackages__",
        "@neptune//:__subpackages__",
    ],
    deps = [
        "//third_party/py/etils/epath",
        "@jax",
        "@jaxtyping",
        "@orbax//:checkpoint",
    ],
)
