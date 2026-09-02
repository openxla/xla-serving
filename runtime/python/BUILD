load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@rules_python//python:packaging.bzl", "py_wheel")
load("//:version.bzl", "VERSION")

package_group(
    name = "friends",
    packages = [],
)

package_group(
    name = "internal",
    packages = [],
)

package(default_visibility = [
    ":internal",
])

pybind_extension(
    name = "model_runner",
    srcs = ["model_runner.cc"],
    deps = [
        ":ifrt_client_factory",
        ":numpy_utils",
        "//examples/composite_processor:composite_processor_example_support",
        "//runtime/neptune_data:interpreter_composite_processor_support",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@nanobind_bazel//:nanobind",
        "@neptune//runtime/core:device_buffer",
        "@neptune//runtime/core:host_buffer",
        "@neptune//runtime/core:value",
        "@neptune//runtime/interpreter",
        "@neptune//runtime/interpreter:run_options",
        "@neptune//runtime/kernel:kernel_io_data_type",
        "@neptune//runtime/tensorflow:interpreter_tensorflow_support",  # buildcleaner: keep
        "@neptune//runtime/tensorflow:tf_compatible_interpreter_create_options",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "@xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "@xla//xla/python:nb_numpy",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/python/lib/core:numpy",
    ],
)

config_setting(
    name = "linux_x86_64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
)

config_setting(
    name = "linux_aarch64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:aarch64",
    ],
)

genrule(
    name = "stage_neptune_model_runner_pkg",
    srcs = [
        "__init__.py",
        ":model_runner.so",
    ],
    outs = [
        "neptune_model_runner/__init__.py",
        "neptune_model_runner/model_runner.so",
    ],
    cmd = """
  mkdir -p $(@D)/neptune_model_runner
  cp $(location __init__.py) $(@D)/neptune_model_runner/__init__.py
  cp $(location model_runner.so) $(@D)/neptune_model_runner/model_runner.so
  """,
)

py_library(
    name = "neptune_model_runner_pkg",
    srcs = [
        "neptune_model_runner/__init__.py",
    ],
    data = ["neptune_model_runner/model_runner.so"],
)

py_wheel(
    name = "neptune_model_runner",
    abi = select({
        "@rules_python//python/config_settings:is_python_3.11": "cp311",
        "@rules_python//python/config_settings:is_python_3.12": "abi3",
    }),
    distribution = "neptune_model_runner",
    platform = select({
        ":linux_x86_64": "manylinux_2_28_x86_64",
        ":linux_aarch64": "manylinux_2_28_aarch64",
    }),
    python_tag = select({
        "@rules_python//python/config_settings:is_python_3.11": "cp311",
        "@rules_python//python/config_settings:is_python_3.12": "cp312",
    }),
    strip_path_prefixes = ["runtime/python"],
    version = VERSION,
    deps = [
        ":neptune_model_runner_pkg",
        ":stage_neptune_model_runner_pkg",
    ],
)

cc_library(
    name = "numpy_utils",
    srcs = ["numpy_utils.cc"],
    hdrs = ["numpy_utils.h"],
    copts = ["-fexceptions"],  # go/cpp-3p-exceptions
    features = ["-use_header_modules"],  # Required for -fexceptions.
    deps = [
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
        "@nanobind_bazel//:nanobind",
        "@neptune//runtime/core:allocator",
        "@neptune//runtime/core:default_allocator",
        "@neptune//runtime/core:host_buffer",
        "@neptune//runtime/tensorflow/utils:ifrt_tensor_utils",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//third_party/py/numpy:headers",
        "@tsl//tsl/platform:tstring",
        "@xla//third_party/python_runtime:headers",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/python:nb_numpy",
        "@xla//xla/python:types",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/python/lib/core:numpy",
    ],
)

cc_library(
    name = "ifrt_client_factory",
    srcs = ["ifrt_client_factory.cc"],
    hdrs = ["ifrt_client_factory.h"],
    visibility = [":internal"],
    deps = [
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/functional:any_invocable",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@xla//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "@xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt",
        "@xla//xla/tsl/platform:statusor",
    ],
)
