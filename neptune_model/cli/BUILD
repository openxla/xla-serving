load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//storage/binfs:binfs.bzl", "pytype_strict_binary_par")
load("@rules_python//python:proto.bzl", "py_proto_library")

package(default_visibility = ["@neptune//neptune_model/cli:__subpackages__"])

pytype_strict_binary_par(
    name = "npt-cli",
    srcs = ["npt_cli.py"],
    main = "npt_cli.py",
    deps = [
        ":constants",
        ":show",
        "//blobstore2/public:blob_file_register",
        "//file/tfhub",
        "//third_party/py/absl:app",
        "//third_party/py/rich",
        "//third_party/py/typer",
    ],
)

py_library(
    name = "printer",
    srcs = ["printer.py"],
    deps = [
        ":constants",
        "@com_google_protobuf//:protobuf",
        "@neptune//protos:manifest_py_pb2",
        "@neptune//protos:orchestration_py_pb2",
        "@neptune//protos:type_py_pb2",
        "@neptune//protos/xla:xla_vendored_py_pb2",
    ],
)

py_test(
    name = "printer_test",
    srcs = ["printer_test.py"],
    deps = [
        ":printer",
        ":redaction_test_py_pb2",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@neptune//protos:manifest_py_pb2",
        "@neptune//protos:type_py_pb2",
        "@neptune//protos/xla:xla_vendored_py_pb2",
    ],
)

py_library(
    name = "show",
    srcs = ["show.py"],
    deps = [
        ":constants",
        ":printer",
        "//third_party/py/jax/extend",
        "//third_party/py/rich",
        "//third_party/py/typer",
        "//third_party/py/typing_extensions",
        "@com_google_protobuf//:protobuf",
        "@neptune//neptune_model/_src/core:file_utils",
        "@neptune//neptune_model/_src/core:persistence_lib",
        "@neptune//protos:manifest_py_pb2",
        "@neptune//protos:orchestration_py_pb2",
        "@neptune//protos:tf_concrete_function_handle_py_pb2",
        "@neptune_pypi//jax",
    ],
)

py_test(
    name = "show_test",
    srcs = ["show_test.py"],
    deps = [
        ":show",
        "//third_party/py/typer",
        "@absl_py//absl/testing:absltest",
        "@neptune//protos:manifest_py_pb2",
        "@neptune//protos:type_py_pb2",
    ],
)

py_library(
    name = "constants",
    srcs = ["constants.py"],
)

proto_library(
    name = "redaction_test_proto",
    srcs = ["redaction_test.proto"],
)

py_proto_library(
    name = "redaction_test_py_pb2",
    deps = [":redaction_test_proto"],
)
