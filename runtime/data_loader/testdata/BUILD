load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@xla_serving//runtime/data_loader/testdata:gen_checkpoint.bzl", "gen_checkpoint")

package(default_visibility = [
    "@xla_serving//:__subpackages__",
])

gen_checkpoint(
    name = "non_spmd_ocdbt",
    args = [
        "--enable_sharding=false",
        "--serialization_method=ocdbt",
    ],
    env = {
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
    },
    exec_tool = ":gen_checkpoint",
)

gen_checkpoint(
    name = "spmd_ocdbt",
    args = [
        "--enable_sharding=true",
        "--serialization_method=ocdbt",
    ],
    env = {
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
    },
    exec_tool = ":gen_checkpoint",
)

gen_checkpoint(
    name = "non_spmd_checkpoint",
    args = [
        "--enable_sharding=false",
        "--serialization_method=jax",
    ],
    env = {
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
    },
    exec_tool = ":gen_checkpoint",
)

gen_checkpoint(
    name = "spmd_checkpoint",
    args = [
        "--enable_sharding=true",
        "--serialization_method=jax",
    ],
    env = {
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
    },
    exec_tool = ":gen_checkpoint",
)

py_binary(
    name = "gen_checkpoint",
    srcs = ["gen_checkpoint.py"],
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@xla_serving_pypi//jax",
        "@xla_serving_pypi//jaxtyping",
        "@xla_serving_pypi//numpy",
        "@xla_serving_pypi//orbax_checkpoint",
    ],
)

bzl_library(
    name = "gen_checkpoint_bzl",
    srcs = ["gen_checkpoint.bzl"],
    visibility = ["//visibility:private"],
)
