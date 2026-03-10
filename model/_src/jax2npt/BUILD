load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library", pytype_strict_test = "pytype_strict_contrib_test")

package(default_visibility = ["@neptune//model:__subpackages__"])

pytype_strict_library(
    name = "typing",
    srcs = ["typing.py"],
    deps = ["//third_party/py/jaxtyping"],
)

pytype_strict_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        ":typing",
        "//third_party/py/jax",
        "//third_party/py/jax/experimental:layout",
        "//third_party/py/jax/extend:sharding",
        "//third_party/py/jaxtyping",
        "//third_party/py/numpy",
        "@neptune//model/_src/core:shlo_type",
        "@neptune//protos:manifest_py_pb2",
        "@xla//xla:xla_data_proto_py_pb2",
    ],
)

pytype_strict_library(
    name = "main_lib",
    srcs = ["main_lib.py"],
    deps = [
        ":constants",
        ":typing",
        ":utils",
        "//third_party/py/jax",
        "//third_party/py/jax/experimental:layout",
        "//third_party/py/numpy",
        "//third_party/py/orbax/checkpoint",
        "@neptune//model/_src/core:shlo_function",
        "@neptune//model/_src/core:unstructured_data",
        "@neptune//model/_src/core:value",
        "@neptune//protos:manifest_py_pb2",
    ],
)

pytype_strict_library(
    name = "constants",
    srcs = ["constants.py"],
)

pytype_strict_library(
    name = "__init__",
    srcs = ["__init__.py"],
    deps = [
        ":constants",
        ":main_lib",
        ":typing",
    ],
)

pytype_strict_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    env = {
        "XLA_FLAGS": "--xla_force_host_platform_device_count=8",
    },
    deps = [
        ":utils",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/jax",
        "//third_party/py/jax/experimental:layout",
        "//third_party/py/numpy",
        "//third_party/py/orbax/experimental/model/core",
        "@neptune//model/_src/core:shlo_type",
        "@xla//xla:xla_data_proto_py_pb2",
    ],
)

pytype_strict_test(
    name = "main_lib_test",
    srcs = ["main_lib_test.py"],
    data = ["@neptune//model/_src/test_utils:mnist_jax_model_expected_manifest.textproto"],
    shard_count = 4,
    deps = [
        ":main_lib",
        "//net/proto2/contrib/pyutil:compare",
        "//net/proto2/python/public",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/jax",
        "//third_party/py/jax/experimental:mesh_utils",
        "//third_party/py/orbax/checkpoint",
        "@neptune//model/_src/core:persistence_lib",
        "@neptune//model/_src/test_utils:jax_mnist",
        "@neptune//protos:manifest_py_pb2",
    ],
)
