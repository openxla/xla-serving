load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//third_party/neptune/model:__subpackages__"])

py_library(
    name = "typing",
    srcs = ["typing.py"],
    deps = ["@jaxtyping"],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        ":typing",
        "//third_party/neptune/model/_src/core:shlo_type",
        "//third_party/neptune/protos:manifest_py_pb2",
        "@jax",
        "@jax//:experimental_layout",
        "@jax//:extend_sharding",
        "@jaxtyping",
        "@numpy",
        "@xla//xla:xla_data_proto_py_pb2",
    ],
)

py_library(
    name = "main_lib",
    srcs = ["main_lib.py"],
    deps = [
        ":constants",
        ":typing",
        ":utils",
        "//third_party/neptune/model/_src/core:shlo_function",
        "//third_party/neptune/model/_src/core:unstructured_data",
        "//third_party/neptune/model/_src/core:value",
        "//third_party/neptune/protos:manifest_py_pb2",
        "@jax",
        "@jax//:experimental_layout",
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

pytype_strict_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    env = {
        "XLA_FLAGS": "--xla_force_host_platform_device_count=8",
    },
    deps = [
        ":utils",
        "//third_party/neptune/model/_src/core:shlo_type",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jax",
        "@jax//:experimental_layout",
        "@numpy",
        "@orbax//:experimental_model_core",
        "@xla//xla:xla_data_proto_py_pb2",
    ],
)

pytype_strict_test(
    name = "main_lib_test",
    srcs = ["main_lib_test.py"],
    data = ["//third_party/neptune/model/_src/test_utils:mnist_jax_model_expected_manifest.textproto"],
    shard_count = 4,
    deps = [
        ":main_lib",
        "//third_party/neptune/model/_src/core:persistence_lib",
        "//third_party/neptune/model/_src/test_utils:jax_mnist",
        "//third_party/neptune/protos:manifest_py_pb2",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jax",
        "@jax//:experimental_mesh_utils",
        "@orbax//:checkpoint",
    ],
)
