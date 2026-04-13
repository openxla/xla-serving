load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//third_party/neptune/model:__subpackages__"])

py_library(
    name = "main_lib",
    srcs = ["main_lib.py"],
    deps = [
        ":constants",
        ":jax_model",
        ":serving_config",
        "//third_party/neptune/model/_src/core:orchestration_proto_utils",
        "//third_party/neptune/model/_src/core:persistence_lib",
        "//third_party/neptune/model/_src/core:saveable",
        "//third_party/neptune/model/_src/core:shlo_type",
        "//third_party/neptune/model/_src/core:type_proto_utils",
        "//third_party/neptune/model/_src/jax2npt:__init__",
        "//third_party/neptune/protos:orchestration_py_pb2",
        "@com_google_absl_py//absl/logging",
        "@jax",
        "@jaxtyping",
        "@orbax//:checkpoint",
    ],
)

py_library(
    name = "jax_model",
    srcs = ["jax_model.py"],
    deps = [
        ":constants",
        ":utils",
        "//third_party/neptune/model/_src/core:compile_options_util",
        "//third_party/neptune/model/_src/core:shlo_function",
        "//third_party/neptune/model/_src/core:shlo_type",
        "//third_party/neptune/model/_src/jax2npt:__init__",
        "//third_party/neptune/model/_src/jax2npt:typing",
        "@com_google_absl_py//absl/logging",
        "@jax",
        "@jaxtyping",
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
        "//third_party/neptune/protos:orchestration_py_pb2",
        "@com_google_absl_py//absl/logging",
        "@jaxtyping",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "@jax",
        "@jaxtyping",
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
        "//third_party/neptune/protos:orchestration_py_pb2",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_test(
    name = "main_lib_test",
    srcs = ["main_lib_test.py"],
    deps = [
        ":jax_model",
        ":main_lib",
        ":serving_config",
        "//third_party/neptune/model/_src/core:persistence_lib",
        "//third_party/neptune/model/_src/core:saveable",
        "//third_party/neptune/model/_src/core:shlo_type",
        "//third_party/neptune/protos:orchestration_py_pb2",
        "@com_google_absl_py//absl/testing:absltest",
        "@jax",
    ],
)

pytype_strict_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    deps = [
        ":jax_model",
        ":utils",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jax",
    ],
)
