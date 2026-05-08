load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    "@neptune//examples:__subpackages__",
])

py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        "//third_party/py/flax/nnx",
        "@neptune_pypi//jax",
    ],
)

py_library(
    name = "export",
    srcs = ["export.py"],
    deps = [
        ":model",
        "//third_party/py/flax/nnx",
        "@absl_py//absl/logging",
        "@neptune//neptune_model/_src/export:jax_model",
        "@neptune//neptune_model/_src/export:main_lib",
        "@neptune//neptune_model/_src/export:serving_config",
        "@neptune_pypi//jax",
        "@neptune_pypi//orbax_checkpoint",
    ],
)

py_binary(
    name = "run_export",
    srcs = ["run_export.py"],
    deps = [
        ":export",
        "//third_party/py/absl:app",
        "@absl_py//absl/flags",
    ],
)

py_test(
    name = "export_test",
    srcs = ["export_test.py"],
    deps = [
        ":export",
        "@absl_py//absl/testing:absltest",
    ],
)

cc_binary(
    name = "client",
    srcs = ["client.cc"],
    deps = [
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/flags:parse",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@com_github_grpc_grpc//:grpc++",
        "@neptune//runtime/core:host_buffer",
        "@neptune//runtime/core/proto:host_buffer_cc_proto",
        "@neptune//runtime/serving/gcp/auth:utils",
        "@neptune//runtime/serving/gcp/proto:model_cc_proto",
        "@neptune//runtime/serving/gcp/proto:predict_cc_proto",
        "@neptune//runtime/serving/gcp/proto:prediction_log_cc_proto",
        "@neptune//runtime/serving/gcp/proto:prediction_service_cc_grpc_proto",
        "@xla//xla/python/ifrt:dtype_proto_cc",
        "@xla//xla/python/ifrt:shape_proto_cc",
        "@xla//xla/tsl/platform:statusor",
    ],
)
