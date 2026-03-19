load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_testonly = 1,
    default_visibility = ["//third_party/neptune/model:__subpackages__"],
)

exports_files(
    ["mnist_jax_model_expected_manifest.textproto"],
)

py_library(
    name = "jax_mnist",
    srcs = ["jax_mnist.py"],
    deps = [
        "//third_party/py/flax:core",
    ],
)
