load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")
load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = [
    "@xla_serving//:__subpackages__",
])

py_library(
    name = "transform",
    srcs = ["transform.py"],
)

pybind_extension(
    name = "composite_processor_pybind",
    srcs = ["composite_processor_pybind.cc"],
    dynamic_deps = ["@xla_serving//data:neptune_data_core"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../.."],
    deps = [
        ":transform_pybind",
        "@abseil-cpp//absl/base:fast_type_id",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@com_google_protobuf//:any_cc_proto",
        "@dlpack",
        "@nanobind_bazel//:nanobind",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
        "@tsl//tsl/platform:tstring",
        "@xla//xla/python:types",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/platform:status_macros",
        "@xla_serving//data/processor/cc/api:transform_context",
        "@xla_serving//data/processor/cc/api:type_info",
        "@xla_serving//data/processor/cc/api:value",
        "@xla_serving//data/processor/cc/core:composite_processor",
        "@xla_serving//data/processor/cc/core:transform",
        "@xla_serving//data/processor/cc/core/graph:execution_graph",
        "@xla_serving//data/processor/cc/utils:buffer_utils",
        "@xla_serving//data/processor/proto:transform_config_cc_proto",
        "@xla_serving//data/resource_manager/proto:resource_manager_cc_proto",
        "@xla_serving//runtime/core:host_buffer",
    ],
)

pybind_extension(
    name = "composite_processor_test_utils_pybind",
    testonly = True,
    srcs = ["composite_processor_test_utils_pybind.cc"],
    deps = [
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
        "@xla_serving//data/processor/cc/core:transform",
        "@xla_serving//data/processor/cc/utils:buffer_utils",
        "@xla_serving//runtime/core:host_buffer",
    ],
)

py_test(
    name = "composite_processor_test",
    srcs = ["composite_processor_test.py"],
    deps = [
        ":composite_processor_pybind",
        ":composite_processor_test_utils_pybind",
        "@absl_py//absl/testing:absltest",
        "@xla_serving//data/processor/proto:transform_config_py_pb2",
        "@xla_serving//data/processor/python/examples:add_from_resource_transform_pybind",
        "@xla_serving//data/processor/python/examples:add_host_buffer_transform_pybind",
        "@xla_serving//data/processor/python/examples:add_int_transform_pybind",
        "@xla_serving//data/processor/python/examples:add_two_transform_pybind",
        "@xla_serving//data/processor/python/examples:int_resource_pybind",
        "@xla_serving//data/processor/python/examples:make_dict_transform_pybind",
        "@xla_serving//data/processor/python/examples:multiply_by_five_transform_pybind",
        "@xla_serving//data/processor/python/examples:suffix_transform_pybind",
        "@xla_serving//data/resource_manager/python:resource_manager",
        "@xla_serving_pypi//jax",
        "@xla_serving_pypi//ml_dtypes",  # build_cleaner: keep
        "@xla_serving_pypi//numpy",
    ],
)

pybind_library(
    name = "transform_pybind",
    hdrs = ["transform_pybind.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
        "@xla_serving//data/processor/cc/api:transform_context",
        "@xla_serving//data/processor/cc/api:type_info",
        "@xla_serving//data/processor/cc/api:value",
        "@xla_serving//data/processor/cc/core:transform",
        "@xla_serving//data/processor/proto:transform_config_cc_proto",
    ],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
)

py_binary(
    name = "generate_pybind",
    srcs = ["generate_pybind.py"],
)

py_test(
    name = "generate_pybind_test",
    srcs = [
        "generate_pybind.py",
        "generate_pybind_test.py",
    ],
    deps = ["@absl_py//absl/testing:absltest"],
)
