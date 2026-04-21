load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@rules_python//python:py_test.bzl", "py_test")

package(default_visibility = ["//third_party/neptune:__subpackages__"])

py_library(
    name = "processor",
    srcs = ["processor.py"],
    deps = [
        "//third_party/neptune/model/_src/core:saveable",
        "@jaxtyping",
    ],
)

py_library(
    name = "jax_processor",
    srcs = ["jax_processor.py"],
    deps = [
        "//third_party/neptune/model/_src/core:saveable",
        "//third_party/neptune/model/_src/core:shlo_function",
        "//third_party/neptune/model/_src/core:shlo_type",
        "//third_party/neptune/model/_src/export:constants",
        "//third_party/neptune/model/_src/export:jax_model",
        "//third_party/neptune/model/_src/export:utils",
        "@jax",
        "@jaxtyping",
    ],
)

py_test(
    name = "jax_processor_test",
    srcs = ["jax_processor_test.py"],
    deps = [
        ":jax_processor",
        "//third_party/neptune/model/_src/core:shlo_type",
        "//third_party/neptune/model/_src/export:jax_model",
        "@com_google_absl_py//absl/testing:absltest",
        "@jax",
    ],
)

py_library(
    name = "tf_processor",
    srcs = ["tf_processor.py"],
    deps = [
        "//third_party/neptune/model/_src/core:saveable",
        "//third_party/neptune/model/_src/tf2npt:__init__",
        "@jaxtyping",
    ],
)

py_test(
    name = "tf_processor_test",
    srcs = ["tf_processor_test.py"],
    deps = [
        ":tf_processor",
        "//third_party/neptune/model/_src/core:shlo_type",
        "//third_party/neptune/model/_src/core:test_utils",
        "//third_party/neptune/model/_src/tf2npt:converter",
        "//third_party/neptune/protos:tf_concrete_function_handle_py_pb2",
        "//third_party/py/tensorflow",
        "@com_google_absl_py//absl/testing:absltest",
    ],
)
