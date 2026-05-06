load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@rules_python//python:py_test.bzl", "py_test")

package(default_visibility = ["@neptune//:__subpackages__"])

py_library(
    name = "processor",
    srcs = ["processor.py"],
    deps = [
        "@neptune//model/_src/core:saveable",
        "@neptune//model/_src/core:value",
        "@neptune_pypi//jaxtyping",
    ],
)

py_library(
    name = "jax_processor",
    srcs = ["jax_processor.py"],
    deps = [
        "@neptune//model/_src/core:saveable",
        "@neptune//model/_src/core:shlo_function",
        "@neptune//model/_src/core:shlo_type",
        "@neptune//model/_src/core:value",
        "@neptune//model/_src/export:constants",
        "@neptune//model/_src/export:jax_model",
        "@neptune//model/_src/export:utils",
        "@neptune//model/_src/jax2npt:__init__",
        "@neptune_pypi//jax",
        "@neptune_pypi//jaxtyping",
        "@neptune_pypi//orbax_checkpoint",
    ],
)

py_test(
    name = "jax_processor_test",
    srcs = ["jax_processor_test.py"],
    deps = [
        ":jax_processor",
        "@absl_py//absl/testing:absltest",
        "@neptune//model/_src/core:shlo_type",
        "@neptune//model/_src/export:jax_model",
        "@neptune_pypi//jax",
    ],
)

py_library(
    name = "tf_processor",
    srcs = ["tf_processor.py"],
    deps = [
        "@neptune//model/_src/core:saveable",
        "@neptune//model/_src/core:value",
        "@neptune//model/_src/tf2npt:__init__",
        "@neptune_pypi//jaxtyping",
    ],
)

py_test(
    name = "tf_processor_test",
    srcs = ["tf_processor_test.py"],
    deps = [
        ":tf_processor",
        "//third_party/py/tensorflow",
        "@absl_py//absl/testing:absltest",
        "@neptune//model/_src/core:shlo_type",
        "@neptune//model/_src/core:test_utils",
        "@neptune//model/_src/tf2npt:converter",
        "@neptune//protos:tf_concrete_function_handle_py_pb2",
    ],
)
