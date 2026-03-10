load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library", pytype_strict_test = "pytype_strict_contrib_test")
load("//devtools/python/blaze:strict.bzl", "py_strict_test")

package(default_visibility = ["@neptune//model:__subpackages__"])

pytype_strict_library(
    name = "function",
    srcs = ["function.py"],
    deps = [
        ":shlo_type",
        "//third_party/py/jaxtyping",
    ],
)

pytype_strict_test(
    name = "unstructured_data_test",
    srcs = ["unstructured_data_test.py"],
    deps = [
        ":unstructured_data",
        "//third_party/py/absl/testing:absltest",
        "@neptune//protos:manifest_py_pb2",
    ],
)

pytype_strict_test(
    name = "type_proto_utils_test",
    srcs = ["type_proto_utils_test.py"],
    deps = [
        ":shlo_type",
        ":test_utils",
        ":type_proto_utils",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/google/protobuf",
        "@neptune//protos:type_py_pb2",
        "@xla//xla:xla_data_proto_py_pb2",
    ],
)

pytype_strict_test(
    name = "device_assignment_test",
    srcs = ["device_assignment_test.py"],
    deps = [
        ":device_assignment",
        "//learning/brain/research/jax:tpu_support",
        "//testing/pybase:parameterized",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/jax",
        "//third_party/py/jax/experimental:mesh_utils",
        "//third_party/py/jax/experimental:topologies",
        "//third_party/py/numpy",
    ],
)

pytype_strict_test(
    name = "compile_options_util_test",
    srcs = ["compile_options_util_test.py"],
    deps = [
        ":compile_options_util",
        "//platforms/xla/service/jellyfish:tpu_compilation_environment_py_pb2",
        "//platforms/xla/service/jellyfish/python:tpu_compilation_environment",
        "//third_party/py/absl/testing:absltest",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/google/protobuf",
        "//third_party/py/jax",
        "//third_party/py/jax/experimental:mesh_utils",
    ],
)

pytype_strict_library(
    name = "shlo_function",
    srcs = ["shlo_function.py"],
    deps = [
        ":function",
        ":shlo_type",
        ":unstructured_data",
    ],
)

pytype_strict_library(
    name = "unstructured_data",
    srcs = ["unstructured_data.py"],
    deps = [
        ":file_utils",
        "@neptune//protos:manifest_py_pb2",
    ],
)

pytype_strict_library(
    name = "serializable_function",
    srcs = ["serializable_function.py"],
    deps = [
        ":function",
        ":unstructured_data",
    ],
)

pytype_strict_library(
    name = "value",
    srcs = ["value.py"],
    deps = [
        ":shlo_type",
        "//third_party/py/jaxtyping",
        "@neptune//protos:manifest_py_pb2",
    ],
)

pytype_strict_library(
    name = "file_utils",
    srcs = ["file_utils.py"],
    deps = [
        "//pyglib:gfile",
        "//pyglib/contrib/gpathlib:gpath",
    ],
)

pytype_strict_library(
    name = "saveable",
    srcs = ["saveable.py"],
    deps = [
        ":function",
        ":value",
    ],
)

pytype_strict_library(
    name = "shlo_type",
    srcs = ["shlo_type.py"],
    deps = [
        "//third_party/py/numpy",
        "@xla//xla:xla_data_proto_py_pb2",
    ],
)

pytype_strict_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    deps = ["//third_party/py/absl/testing:parameterized"],
)

pytype_strict_library(
    name = "type_proto_utils",
    srcs = ["type_proto_utils.py"],
    deps = [
        ":shlo_type",
        "//third_party/py/jax",
        "//third_party/py/jaxtyping",
        "@neptune//protos:type_py_pb2",
    ],
)

pytype_strict_library(
    name = "device_assignment",
    srcs = ["device_assignment.py"],
    deps = ["//third_party/py/jax"],
)

pytype_strict_library(
    name = "compile_options_util",
    srcs = ["compile_options_util.py"],
    deps = [
        "@com_google_protobuf//:protobuf", # any_py_pb2
        "//platforms/xla/service/jellyfish:tpu_compilation_environment_py_pb2",
        "//platforms/xla/service/jellyfish/python:tpu_compilation_environment",
        "//third_party/py/absl/flags",
        "//third_party/py/google/protobuf",
        "//third_party/py/jax",
        "@neptune//protos:manifest_py_pb2",
        "@xla//xla:xla_data_proto_py_pb2",
        "@xla//xla:xla_py_pb2",
        "@xla//xla/pjrt/proto:compile_options_py_pb2",
    ],
)

pytype_strict_library(
    name = "constants",
    srcs = ["constants.py"],
)

pytype_strict_library(
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
        "//third_party/py/absl/logging",
        "@neptune//protos:manifest_py_pb2",
    ],
)

pytype_strict_library(
    name = "metadata",
    srcs = ["metadata.py"],
    deps = [":file_utils"],
)

pytype_strict_library(
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
        "//third_party/py/absl/logging",
        "@neptune//protos:manifest_py_pb2",
    ],
)

py_strict_test(
    name = "manifest_utils_test",
    srcs = ["manifest_utils_test.py"],
    deps = [
        ":device_assignment",
        ":manifest_utils",
        ":serializable_function",
        ":unstructured_data",
        "//third_party/py/absl/testing:absltest",
        "@neptune//protos:manifest_py_pb2",
    ],
)

py_strict_test(
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
        "//net/proto2/contrib/pyutil:compare",
        "//third_party/py/absl/testing:absltest",
        "@neptune//protos:manifest_py_pb2",
        "@neptune//protos:type_py_pb2",
    ],
)
