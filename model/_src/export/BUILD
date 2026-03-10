load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library", pytype_strict_test = "pytype_strict_contrib_test")

package(default_visibility = ["@neptune//model:__subpackages__"])

pytype_strict_library(
    name = "main_lib",
    srcs = ["main_lib.py"],
    deps = [
        ":constants",
        ":jax_model",
        ":serving_config",
        ":utils",
        "//third_party/py/absl/logging",
        "//third_party/py/jax",
        "//third_party/py/jaxtyping",
        "@neptune//model/_src/core:persistence_lib",
        "@neptune//model/_src/core:saveable",
        "@neptune//model/_src/core:shlo_type",
        "@neptune//model/_src/jax2npt:__init__",
        "@neptune//protos:orchestration_py_pb2",
    ],
)

pytype_strict_library(
    name = "jax_model",
    srcs = ["jax_model.py"],
    deps = [
        "//third_party/py/jax",
        "//third_party/py/jaxtyping",
        "@neptune//model/_src/jax2npt:typing",
    ],
)

pytype_strict_test(
    name = "jax_model_test",
    srcs = ["jax_model_test.py"],
    deps = [
        ":jax_model",
        "//third_party/py/absl/testing:absltest",
    ],
)

pytype_strict_library(
    name = "serving_config",
    srcs = ["serving_config.py"],
    deps = ["//third_party/py/jaxtyping"],
)

pytype_strict_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        ":constants",
        ":jax_model",
        "//third_party/py/jax",
        "//third_party/py/jaxtyping",
        "@neptune//model/_src/core:persistence_lib",
        "@neptune//model/_src/jax2npt:typing",
        "@neptune//protos:manifest_py_pb2",
        "@neptune//protos:orchestration_py_pb2",
    ],
)

pytype_strict_library(
    name = "constants",
    srcs = ["constants.py"],
)

pytype_strict_test(
    name = "main_lib_test",
    srcs = ["main_lib_test.py"],
    deps = [
        ":jax_model",
        ":main_lib",
        ":serving_config",
        "//net/proto2/contrib/pyutil:compare",
        "//testing/pybase",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/jax",
        "@neptune//model/_src/core:persistence_lib",
        "@neptune//model/_src/core:saveable",
        "@neptune//model/_src/core:shlo_type",
        "@neptune//protos:orchestration_py_pb2",
    ],
)

pytype_strict_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    deps = [
        ":jax_model",
        ":utils",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/jax",
    ],
)
