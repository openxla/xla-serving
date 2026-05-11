load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    "//third_party/py/neptune_model:__subpackages__",
    "@neptune//neptune_model:__subpackages__",
])

py_library(
    name = "function",
    srcs = ["function.py"],
    deps = [
        ":shlo_type",
        "@neptune_pypi//jaxtyping",
    ],
)

py_test(
    name = "unstructured_data_test",
    srcs = ["unstructured_data_test.py"],
    deps = [
        ":unstructured_data",
        "@absl_py//absl/testing:absltest",
        "@neptune//protos:manifest_py_pb2",
    ],
)

py_test(
    name = "type_proto_utils_test",
    srcs = ["type_proto_utils_test.py"],
    deps = [
        ":shlo_type",
        ":test_utils",
        ":type_proto_utils",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@com_google_protobuf//:protobuf",
        "@neptune//protos:type_py_pb2",
        "@neptune//protos/xla:xla_vendored_py_pb2",
    ],
)

py_test(
    name = "device_assignment_test",
    srcs = ["device_assignment_test.py"],
    deps = [
        ":device_assignment",
        "//testing/pybase:parameterized",
        "@absl_py//absl/testing:absltest",
        "@neptune_pypi//jax",
        "@neptune_pypi//numpy",
    ],
)

py_test(
    name = "compile_options_util_test",
    srcs = ["compile_options_util_test.py"],
    deps = [
        ":compile_options_util",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@com_google_protobuf//:protobuf",
        "@neptune_pypi//jax",
    ],
)

py_library(
    name = "shlo_function",
    srcs = ["shlo_function.py"],
    deps = [
        ":function",
        ":shlo_type",
        ":unstructured_data",
    ],
)

py_library(
    name = "unstructured_data",
    srcs = ["unstructured_data.py"],
    deps = [
        ":file_utils",
        "@neptune//protos:manifest_py_pb2",
    ],
)

py_library(
    name = "serializable_function",
    srcs = ["serializable_function.py"],
    deps = [
        ":function",
        ":unstructured_data",
    ],
)

py_library(
    name = "value",
    srcs = ["value.py"],
    deps = [
        ":shlo_type",
        "@neptune//protos:manifest_py_pb2",
        "@neptune_pypi//jaxtyping",
    ],
)

py_library(
    name = "file_utils",
    srcs = ["file_utils.py"],
)

py_library(
    name = "saveable",
    srcs = ["saveable.py"],
    deps = [
        ":function",
        ":value",
    ],
)

py_library(
    name = "shlo_type",
    srcs = ["shlo_type.py"],
    deps = [
        "@neptune//protos/xla:xla_vendored_py_pb2",
        "@neptune_pypi//numpy",
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    deps = [
        "@absl_py//absl/testing:parameterized",
        "@neptune_pypi//jax",
        "@neptune_pypi//jaxtyping",
    ],
)

py_library(
    name = "type_proto_utils",
    srcs = ["type_proto_utils.py"],
    deps = [
        ":shlo_type",
        "@neptune//protos:type_py_pb2",
        "@neptune_pypi//jax",
        "@neptune_pypi//jaxtyping",
    ],
)

py_library(
    name = "device_assignment",
    srcs = ["device_assignment.py"],
    deps = ["@neptune_pypi//jax"],
)

py_library(
    name = "compile_options_util",
    srcs = ["compile_options_util.py"],
    deps = [
        "@absl_py//absl/flags",
        "@com_google_protobuf//:protobuf",  # any_py_pb2
        "@neptune//protos:manifest_py_pb2",
        "@neptune//protos/xla:xla_vendored_py_pb2",
        "@neptune_pypi//jax",
    ],
)

py_library(
    name = "constants",
    srcs = ["constants.py"],
)

py_library(
    name = "manifest_utils",
    srcs = ["manifest_utils.py"],
    deps = [
        ":constants",
        ":device_assignment",
        ":function",
        ":saveable",
        ":serializable_function",
        ":shlo_function",
        ":type_proto_utils",
        ":unstructured_data",
        ":value",
        "@absl_py//absl/logging",
        "@neptune//protos:manifest_py_pb2",
    ],
)

py_library(
    name = "metadata",
    srcs = ["metadata.py"],
    deps = [":file_utils"],
)

py_library(
    name = "persistence_lib",
    srcs = ["persistence_lib.py"],
    deps = [
        ":constants",
        ":device_assignment",
        ":file_utils",
        ":manifest_utils",
        ":metadata",
        ":saveable",
        ":unstructured_data",
        "@absl_py//absl/logging",
        "@neptune//protos:manifest_py_pb2",
    ],
)

py_test(
    name = "manifest_utils_test",
    srcs = ["manifest_utils_test.py"],
    deps = [
        ":device_assignment",
        ":manifest_utils",
        ":serializable_function",
        ":unstructured_data",
        "@absl_py//absl/testing:absltest",
        "@neptune//protos:manifest_py_pb2",
    ],
)

py_test(
    name = "persistence_lib_test",
    srcs = ["persistence_lib_test.py"],
    deps = [
        ":constants",
        ":device_assignment",
        ":file_utils",
        ":function",
        ":persistence_lib",
        ":serializable_function",
        ":shlo_function",
        ":shlo_type",
        ":unstructured_data",
        ":value",
        "@absl_py//absl/testing:absltest",
        "@neptune//protos:manifest_py_pb2",
        "@neptune//protos:type_py_pb2",
    ],
)

py_library(
    name = "orchestration_proto_utils",
    srcs = ["orchestration_proto_utils.py"],
    deps = [
        ":constants",
        ":shlo_type",
        ":type_proto_utils",
        "@neptune//protos:manifest_py_pb2",
        "@neptune//protos:orchestration_py_pb2",
        "@neptune_pypi//jax",
        "@neptune_pypi//jaxtyping",
    ],
)

py_test(
    name = "orchestration_proto_utils_test",
    srcs = ["orchestration_proto_utils_test.py"],
    deps = [
        ":orchestration_proto_utils",
        ":shlo_type",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)
