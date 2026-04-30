load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    "@neptune//examples:__subpackages__",
    "@neptune//model:__subpackages__",
])

py_library(
    name = "main_lib",
    srcs = ["main_lib.py"],
    deps = [
        ":constants",
        ":jax_model",
        ":serving_config",
        ":utils",
        "@com_google_absl_py//absl/logging",
        "@jax",
        "@jaxtyping",
        "@neptune//model/_src/core:orchestration_proto_utils",
        "@neptune//model/_src/core:persistence_lib",
        "@neptune//model/_src/core:saveable",
        "@neptune//model/_src/core:shlo_type",
        "@neptune//model/_src/core:type_proto_utils",
        "@neptune//model/_src/core:value",
        "@neptune//model/_src/processors:jax_processor",
        "@neptune//model/_src/processors:processor",
        "@neptune//model/_src/processors:tf_processor",
        "@neptune//model/_src/tf2npt:__init__",
        "@neptune//protos:orchestration_py_pb2",
    ],
)

py_library(
    name = "jax_model",
    srcs = ["jax_model.py"],
    deps = [
        ":constants",
        ":utils",
        "@com_google_absl_py//absl/logging",
        "@jax",
        "@jaxtyping",
        "@neptune//model/_src/core:compile_options_util",
        "@neptune//model/_src/core:shlo_function",
        "@neptune//model/_src/core:shlo_type",
        "@neptune//model/_src/jax2npt:__init__",
        "@neptune//model/_src/jax2npt:typing",
    ],
)

pytype_strict_test(
    name = "jax_model_test",
    srcs = ["jax_model_test.py"],
    deps = [
        ":jax_model",
        "@com_google_absl_py//absl/testing:absltest",
        "@jax",
    ],
)

py_library(
    name = "serving_config",
    srcs = ["serving_config.py"],
    deps = [
        "@com_google_absl_py//absl/logging",
        "@jaxtyping",
        "@neptune//model/_src/processors:processor",
        "@neptune//protos:orchestration_py_pb2",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "@jax",
        "@jaxtyping",
        "@neptune//model/_src/core:shlo_type",
        "@numpy",
    ],
)

py_library(
    name = "constants",
    srcs = ["constants.py"],
)

pytype_strict_test(
    name = "serving_config_test",
    srcs = ["serving_config_test.py"],
    deps = [
        ":serving_config",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@neptune//protos:orchestration_py_pb2",
    ],
)

pytype_strict_test(
    name = "main_lib_test",
    srcs = ["main_lib_test.py"],
    deps = [
        ":jax_model",
        ":main_lib",
        ":serving_config",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jax",
        "@neptune//model/_src/core:persistence_lib",
        "@neptune//model/_src/processors:processor",
        "@neptune//protos:orchestration_py_pb2",
    ],
)

pytype_strict_test(
    name = "main_lib_orchestration_test",
    srcs = ["main_lib_orchestration_test.py"],
    data = glob(["testdata/**/*"]),
    deps = [
        ":jax_model",
        ":main_lib",
        ":serving_config",
        "//third_party/py/tensorflow",
        "@com_google_absl_py//absl/testing:absltest",
        "@jax",
        "@neptune//model/_src/core:persistence_lib",
        "@neptune//model/_src/processors:processor",
        "@neptune//model/_src/processors:tf_processor",
        "@neptune//protos:orchestration_py_pb2",
        "@neptune//protos:tf_concrete_function_handle_py_pb2",
    ],
)

pytype_strict_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    deps = [
        ":utils",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jax",
        "@neptune//model/_src/core:shlo_type",
    ],
)
