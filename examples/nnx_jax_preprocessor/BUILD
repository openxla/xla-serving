load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    "@neptune//examples:__subpackages__",
])

py_library(
    name = "export",
    srcs = ["export.py"],
    deps = [
        "//third_party/py/flax/nnx",
        "@neptune//examples/nnx:model",
        "@neptune//model/_src/export:jax_model",
        "@neptune//model/_src/export:main_lib",
        "@neptune//model/_src/export:serving_config",
        "@neptune//model/_src/processors:jax_processor",
        "@neptune//model/_src/processors:processor",
        "@neptune_pypi//jax",
        "@neptune_pypi//orbax_checkpoint",
    ],
)

py_binary(
    name = "run_export",
    srcs = ["run_export.py"],
    deps = [
        ":export",
        "//third_party/py/absl:app",
        "@absl_py//absl/flags",
    ],
)

py_test(
    name = "export_test",
    srcs = ["export_test.py"],
    deps = [
        ":export",
        "@absl_py//absl/testing:absltest",
    ],
)
