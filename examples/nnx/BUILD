load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    "@neptune//examples:__subpackages__",
])

py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        "//third_party/py/flax/nnx",
        "@jax",
    ],
)

py_library(
    name = "export",
    srcs = ["export.py"],
    deps = [
        ":model",
        "//third_party/py/flax/nnx",
        "@com_google_absl_py//absl/logging",
        "@jax",
        "@neptune//model/_src/export:jax_model",
        "@neptune//model/_src/export:main_lib",
        "@neptune//model/_src/export:serving_config",
        "@orbax//:checkpoint",
    ],
)

pytype_strict_binary(
    name = "run_export",
    srcs = ["run_export.py"],
    deps = [
        ":export",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
    ],
)

py_test(
    name = "export_test",
    srcs = ["export_test.py"],
    deps = [
        ":export",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)
