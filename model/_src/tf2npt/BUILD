load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["@neptune//:__subpackages__"])

py_library(
    name = "utils",
    srcs = ["utils.py"],
)

py_library(
    name = "x64_helpers",
    srcs = ["x64_helpers.py"],
)

py_library(
    name = "type_conversion",
    srcs = ["type_conversion.py"],
    deps = [
        "//third_party/py/tensorflow",
        "@neptune//model/_src/core:shlo_type",
        "@numpy",
    ],
)

py_library(
    name = "converter",
    srcs = ["converter.py"],
    deps = [
        ":type_conversion",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/core:protos_all_py_pb2",
        "@jax",
        "@jaxtyping",
        "@neptune//model/_src/core:file_utils",
        "@neptune//model/_src/core:persistence_lib",
        "@neptune//model/_src/core:serializable_function",
        "@neptune//model/_src/core:shlo_type",
        "@neptune//model/_src/core:unstructured_data",
        "@neptune//protos:manifest_py_pb2",
        "@neptune//protos:tf_concrete_function_handle_py_pb2",
        "@numpy",
    ],
)

py_test(
    name = "type_conversion_test",
    srcs = ["type_conversion_test.py"],
    deps = [
        ":type_conversion",
        "//third_party/py/tensorflow",
        "@com_google_absl_py//absl/testing:absltest",
        "@neptune//model/_src/core:shlo_type",
    ],
)

py_test(
    name = "converter_test",
    srcs = ["converter_test.py"],
    deps = [
        ":converter",
        "//third_party/py/chex",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/core:protos_all_py_pb2",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jax",
        "@neptune//model/_src/core:test_utils",
    ],
)

py_test(
    name = "converter_e2e_test",
    srcs = ["converter_e2e_test.py"],
    data = ["testdata/manifest_with_concrete_function.textproto"],
    deps = [
        ":converter",
        "//third_party/py/chex",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@jax",
        "@jaxtyping",
        "@neptune//model/_src/core:persistence_lib",
        "@neptune//protos:manifest_py_pb2",
        "@neptune//protos:tf_concrete_function_handle_py_pb2",
    ],
)
