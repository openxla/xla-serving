load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(
    default_testonly = 1,
    default_visibility = ["@neptune//model:__subpackages__"],
)

exports_files(
    ["mnist_jax_model_expected_manifest.textproto"],
)

pytype_strict_library(
    name = "jax_mnist",
    srcs = ["jax_mnist.py"],
    deps = [
        "//third_party/py/flax:core",
    ],
)
