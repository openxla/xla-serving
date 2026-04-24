load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//storage/binfs:binfs.bzl", "pytype_strict_binary_par")
load("@rules_python//python:proto.bzl", "py_proto_library")

package(default_visibility = ["@neptune//model/cli:__subpackages__"])

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
        "//third_party/py/google/protobuf",
        "@neptune//protos:manifest_py_pb2",
        "@neptune//protos:orchestration_py_pb2",
        "@neptune//protos:type_py_pb2",
        "@xla//xla/pjrt/proto:compile_options_py_pb2",
    ],
)

py_test(
    name = "printer_test",
    srcs = ["printer_test.py"],
    deps = [
        ":printer",
        ":redaction_test_py_pb2",
        "@com_google_absl_py//absl/testing:absltest",
        "@com_google_absl_py//absl/testing:parameterized",
        "@neptune//protos:manifest_py_pb2",
        "@neptune//protos:type_py_pb2",
        "@xla//xla/pjrt/proto:compile_options_py_pb2",
    ],
)

py_library(
    name = "show",
    srcs = ["show.py"],
    deps = [
        ":constants",
        ":printer",
        "//third_party/py/google/protobuf",
        "//third_party/py/jax/extend",
        "//third_party/py/rich",
        "//third_party/py/typer",
        "//third_party/py/typing_extensions",
        "@jax",
        "@neptune//model/_src/core:file_utils",
        "@neptune//model/_src/core:persistence_lib",
        "@neptune//protos:manifest_py_pb2",
        "@neptune//protos:orchestration_py_pb2",
        "@neptune//protos:tf_concrete_function_handle_py_pb2",
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
