load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["@neptune//:__subpackages__"])

py_library(
    name = "__init__",
    srcs = ["__init__.py"],
    deps = [
        ":converter",
    ],
)

py_library(
    name = "type_conversion",
    srcs = ["type_conversion.py"],
    deps = [
        "//third_party/py/tensorflow",
        "@neptune//neptune_model/_src/core:shlo_type",
        "@neptune_pypi//numpy",
    ],
)

py_library(
    name = "converter",
    srcs = ["converter.py"],
    deps = [
        ":type_conversion",
        "//third_party/py/tensorflow",
        "@neptune//neptune_model/_src/core:file_utils",
        "@neptune//neptune_model/_src/core:persistence_lib",
        "@neptune//neptune_model/_src/core:serializable_function",
        "@neptune//neptune_model/_src/core:shlo_type",
        "@neptune//neptune_model/_src/core:unstructured_data",
        "@neptune//protos:manifest_py_pb2",
        "@neptune//protos:tf_concrete_function_handle_py_pb2",
        "@neptune_pypi//jax",
        "@neptune_pypi//jaxtyping",
        "@neptune_pypi//numpy",
        "@org_tensorflow//tensorflow/core:protos_all_py_pb2",
    ],
)

py_test(
    name = "type_conversion_test",
    srcs = ["type_conversion_test.py"],
    deps = [
        ":type_conversion",
        "//third_party/py/tensorflow",
        "@absl_py//absl/testing:absltest",
        "@neptune//neptune_model/_src/core:shlo_type",
    ],
)

py_test(
    name = "converter_test",
    srcs = ["converter_test.py"],
    deps = [
        ":converter",
        "//third_party/py/chex",
        "//third_party/py/tensorflow",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@neptune//neptune_model/_src/core:test_utils",
        "@neptune_pypi//jax",
        "@org_tensorflow//tensorflow/core:protos_all_py_pb2",
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
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@neptune//neptune_model/_src/core:persistence_lib",
        "@neptune//protos:manifest_py_pb2",
        "@neptune//protos:tf_concrete_function_handle_py_pb2",
        "@neptune_pypi//jax",
        "@neptune_pypi//jaxtyping",
    ],
)
