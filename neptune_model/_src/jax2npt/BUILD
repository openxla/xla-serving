load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    "//third_party/py/neptune_model:__subpackages__",
    "@neptune//neptune_model:__subpackages__",
])

py_library(
    name = "typing",
    srcs = ["typing.py"],
    deps = ["@neptune_pypi//jaxtyping"],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        ":typing",
        "@neptune//neptune_model/_src/core:shlo_type",
        "@neptune//protos:manifest_py_pb2",
        "@neptune//protos/xla:xla_vendored_py_pb2",
        "@neptune_pypi//jax",
        "@neptune_pypi//jaxtyping",
        "@neptune_pypi//numpy",
    ],
)

py_library(
    name = "main_lib",
    srcs = ["main_lib.py"],
    deps = [
        ":constants",
        ":typing",
        ":utils",
        "@neptune//neptune_model/_src/core:shlo_function",
        "@neptune//neptune_model/_src/core:unstructured_data",
        "@neptune//neptune_model/_src/core:value",
        "@neptune//protos:manifest_py_pb2",
        "@neptune_pypi//jax",
    ],
)

py_library(
    name = "constants",
    srcs = ["constants.py"],
)

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
    deps = [
        ":constants",
        ":main_lib",
        ":typing",
    ],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    env = {
        "XLA_FLAGS": "--xla_force_host_platform_device_count=8",
    },
    deps = [
        ":utils",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@neptune//neptune_model/_src/core:shlo_type",
        "@neptune//protos/xla:xla_vendored_py_pb2",
        "@neptune_pypi//jax",
        "@neptune_pypi//numpy",
    ],
)

py_test(
    name = "main_lib_test",
    srcs = ["main_lib_test.py"],
    data = ["@neptune//neptune_model/_src/test_utils:mnist_jax_model_expected_manifest.textproto"],
    shard_count = 4,
    deps = [
        ":main_lib",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@neptune//neptune_model/_src/core:persistence_lib",
        "@neptune//neptune_model/_src/test_utils:jax_mnist",
        "@neptune//protos:manifest_py_pb2",
        "@neptune_pypi//jax",
        "@neptune_pypi//orbax_checkpoint",
    ],
)
