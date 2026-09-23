load("@com_google_protobuf//bazel:cc_proto_library.bzl", "cc_proto_library")
load("@com_google_protobuf//bazel:py_proto_library.bzl", "py_proto_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@xla_serving//data/processor/python:build_defs.bzl", "transform_pybind_library")
load("@xla_serving//data/resource_manager/python:build_defs.bzl", "resource_pybind_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "safe_int_utils",
    hdrs = ["safe_int_utils.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
)

py_library(
    name = "cpu_embedding_module",
    srcs = ["cpu_embedding_module.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":cpu_embedding_lookup_transform_pybind",
        ":cpu_embedding_py_pb2",
        ":cpu_embedding_resource_pybind",
        ":cpu_embedding_serializer_pybind",
        "@com_google_protobuf//:protobuf",
        "@xla_serving//data/processor/python:composite_processor_pybind",
        "@xla_serving//data/resource_manager/python:resource_manager",
        "@xla_serving_pypi//jax",
        "@xla_serving_pypi//jaxtyping",
        "@xla_serving_pypi//numpy",
    ],
)

cc_library(
    name = "cpu_embedding_resource",
    srcs = ["cpu_embedding_resource.cc"],
    hdrs = ["cpu_embedding_resource.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        ":cpu_embedding_cc_proto",
        ":safe_int_utils",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:status_macros",
        "@xla_serving//data/cpu_embedding/cc:array",
        "@xla_serving//data/cpu_embedding/cc:cpu_embedding_engine",
        "@xla_serving//data/cpu_embedding/cc:embedding_table",
        "@xla_serving//data/cpu_embedding/cc:feature_spec",
        "@xla_serving//data/resource_manager/cc:asset",
        "@xla_serving//data/resource_manager/cc:resource",
    ],
    alwayslink = 1,
)

cc_test(
    name = "cpu_embedding_resource_test",
    srcs = ["cpu_embedding_resource_test.cc"],
    defines = [
        "ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS",
        "ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS",
    ],
    deps = [
        ":cpu_embedding_cc_proto",
        ":cpu_embedding_resource",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
        "@tsl//tsl/platform:path",
        "@xla//xla/tsl/platform:env",
        "@xla_serving//data/cpu_embedding/cc:cpu_embedding_engine",
        "@xla_serving//data/resource_manager/cc:asset",
        "@xla_serving//data/resource_manager/cc:resource_manager",
        "@xla_serving//data/test_utils:status_macros",
    ],
)

proto_library(
    name = "cpu_embedding_proto",
    srcs = ["cpu_embedding.proto"],
)

cc_proto_library(
    name = "cpu_embedding_cc_proto",
    deps = [":cpu_embedding_proto"],
)

py_proto_library(
    name = "cpu_embedding_py_pb2",
    deps = [":cpu_embedding_proto"],
)

proto_library(
    name = "dense_cpu_embedding_lookup_proto",
    srcs = ["dense_cpu_embedding_lookup.proto"],
)

cc_proto_library(
    name = "dense_cpu_embedding_lookup_cc_proto",
    deps = [":dense_cpu_embedding_lookup_proto"],
)

py_proto_library(
    name = "dense_cpu_embedding_lookup_py_pb2",
    deps = [":dense_cpu_embedding_lookup_proto"],
)

cc_library(
    name = "dense_cpu_embedding_lookup_transform",
    srcs = ["dense_cpu_embedding_lookup_transform.cc"],
    hdrs = ["dense_cpu_embedding_lookup_transform.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        ":cpu_embedding_resource",
        ":dense_cpu_embedding_lookup_cc_proto",
        ":safe_int_utils",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/platform:status_macros",
        "@xla_serving//data/cpu_embedding/cc:array",
        "@xla_serving//data/cpu_embedding/cc:feature_batch_csr_input",
        "@xla_serving//data/processor/cc/api:registry",
        "@xla_serving//data/processor/cc/api:transform_context",
        "@xla_serving//data/processor/cc/api:type_info",
        "@xla_serving//data/processor/cc/core:transform",
        "@xla_serving//data/processor/proto:transform_config_cc_proto",
        "@xla_serving//data/resource_manager/cc:resource",
        "@xla_serving//runtime/core:host_buffer",
    ],
    alwayslink = 1,
)

cc_test(
    name = "dense_cpu_embedding_lookup_transform_test",
    srcs = ["dense_cpu_embedding_lookup_transform_test.cc"],
    defines = [
        "ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS",
        "ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS",
    ],
    deps = [
        ":cpu_embedding_resource",
        ":dense_cpu_embedding_lookup_transform",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
        "@xla//xla/python/ifrt",
        "@xla_serving//data/cpu_embedding/cc:array",
        "@xla_serving//data/cpu_embedding/cc:cpu_embedding_engine",
        "@xla_serving//data/cpu_embedding/cc:embedding_table",
        "@xla_serving//data/cpu_embedding/cc:feature_spec",
        "@xla_serving//data/processor/cc/api:transform_context",
        "@xla_serving//data/processor/cc/core:transform",
        "@xla_serving//data/resource_manager/cc:asset",
        "@xla_serving//data/resource_manager/cc:resource",
        "@xla_serving//data/test_utils:status_macros",
        "@xla_serving//runtime/core:host_buffer",
    ],
)

transform_pybind_library(
    name = "dense_cpu_embedding_lookup_transform_pybind",
    srcs = ["dense_cpu_embedding_lookup_transform.cc"],
    hdrs = ["dense_cpu_embedding_lookup_transform.h"],
    dynamic_deps = ["@xla_serving//data:xla_serving_data_core"],
    generate_wrappers = False,
    linkopts = ["-Wl,-rpath,$$ORIGIN/../.."],
    deps = [
        ":dense_cpu_embedding_lookup_transform",
    ],
)

cc_library(
    name = "cpu_embedding_lookup_transform",
    srcs = ["cpu_embedding_lookup_transform.cc"],
    hdrs = ["cpu_embedding_lookup_transform.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        ":cpu_embedding_cc_proto",
        ":cpu_embedding_resource",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/platform:status_macros",
        "@xla_serving//data/processor/cc/api:registry",
        "@xla_serving//data/processor/cc/api:transform_context",
        "@xla_serving//data/processor/cc/api:type_info",
        "@xla_serving//data/processor/cc/core:transform",
        "@xla_serving//data/processor/proto:transform_config_cc_proto",
        "@xla_serving//data/resource_manager/cc:resource",
        "@xla_serving//runtime/core:host_buffer",
    ],
    alwayslink = 1,
)

cc_test(
    name = "cpu_embedding_lookup_transform_test",
    srcs = ["cpu_embedding_lookup_transform_test.cc"],
    defines = [
        "ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS",
        "ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS",
    ],
    deps = [
        ":cpu_embedding_lookup_transform",
        ":cpu_embedding_resource",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
        "@xla//xla/python/ifrt",
        "@xla_serving//data/cpu_embedding/cc:array",
        "@xla_serving//data/cpu_embedding/cc:cpu_embedding_engine",
        "@xla_serving//data/cpu_embedding/cc:embedding_table",
        "@xla_serving//data/cpu_embedding/cc:feature_spec",
        "@xla_serving//data/processor/cc/api:transform_context",
        "@xla_serving//data/processor/cc/api:value",
        "@xla_serving//data/resource_manager/cc:asset",
        "@xla_serving//data/resource_manager/cc:resource",
        "@xla_serving//data/test_utils:status_macros",
        "@xla_serving//runtime/core:host_buffer",
    ],
)

resource_pybind_library(
    name = "cpu_embedding_resource_pybind",
    srcs = ["cpu_embedding_resource.cc"],
    hdrs = ["cpu_embedding_resource.h"],
    dynamic_deps = ["@xla_serving//data:xla_serving_data_core"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../.."],
    deps = [
        ":cpu_embedding_resource",
    ],
)

transform_pybind_library(
    name = "cpu_embedding_lookup_transform_pybind",
    srcs = ["cpu_embedding_lookup_transform.cc"],
    hdrs = ["cpu_embedding_lookup_transform.h"],
    dynamic_deps = ["@xla_serving//data:xla_serving_data_core"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../.."],
    deps = [
        ":cpu_embedding_lookup_transform",
    ],
)

cc_library(
    name = "cpu_embedding_serializer",
    srcs = ["cpu_embedding_serializer.cc"],
    hdrs = ["cpu_embedding_serializer.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        ":cpu_embedding_resource",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:status_macros",
        "@xla_serving//data/cpu_embedding/cc:array",
        "@xla_serving//data/resource_manager/cc:asset",
    ],
)

pybind_extension(
    name = "cpu_embedding_serializer_pybind",
    srcs = ["cpu_embedding_serializer_pybind.cc"],
    dynamic_deps = ["@xla_serving//data:xla_serving_data_core"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../.."],
    deps = [
        ":cpu_embedding_resource",
        ":cpu_embedding_serializer",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@xla_serving//data/cpu_embedding/python:array_caster",
    ],
)

cc_library(
    name = "pack_host_buffer_vector_transform",
    srcs = ["pack_host_buffer_vector_transform.cc"],
    hdrs = ["pack_host_buffer_vector_transform.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        ":cpu_embedding_cc_proto",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
        "@com_google_protobuf//:any_cc_proto",
        "@xla_serving//data/processor/cc/api:registry",
        "@xla_serving//data/processor/cc/api:transform_context",
        "@xla_serving//data/processor/cc/api:type_info",
        "@xla_serving//data/processor/cc/core:transform",
        "@xla_serving//data/processor/proto:transform_config_cc_proto",
        "@xla_serving//data/resource_manager/cc:resource",
        "@xla_serving//runtime/core:host_buffer",
    ],
    alwayslink = 1,
)

cc_library(
    name = "unpack_host_buffer_vector_transform",
    srcs = ["unpack_host_buffer_vector_transform.cc"],
    hdrs = ["unpack_host_buffer_vector_transform.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        ":cpu_embedding_cc_proto",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@com_google_protobuf//:any_cc_proto",
        "@xla_serving//data/processor/cc/api:registry",
        "@xla_serving//data/processor/cc/api:transform_context",
        "@xla_serving//data/processor/cc/api:type_info",
        "@xla_serving//data/processor/cc/core:transform",
        "@xla_serving//data/processor/proto:transform_config_cc_proto",
        "@xla_serving//data/resource_manager/cc:resource",
        "@xla_serving//runtime/core:host_buffer",
    ],
    alwayslink = 1,
)

transform_pybind_library(
    name = "pack_host_buffer_vector_transform_pybind",
    srcs = ["pack_host_buffer_vector_transform.cc"],
    hdrs = ["pack_host_buffer_vector_transform.h"],
    dynamic_deps = ["@xla_serving//data:xla_serving_data_core"],
    generate_wrappers = False,
    linkopts = ["-Wl,-rpath,$$ORIGIN/../.."],
    deps = [
        ":pack_host_buffer_vector_transform",
    ],
)

transform_pybind_library(
    name = "unpack_host_buffer_vector_transform_pybind",
    srcs = ["unpack_host_buffer_vector_transform.cc"],
    hdrs = ["unpack_host_buffer_vector_transform.h"],
    dynamic_deps = ["@xla_serving//data:xla_serving_data_core"],
    generate_wrappers = False,
    linkopts = ["-Wl,-rpath,$$ORIGIN/../.."],
    deps = [
        ":unpack_host_buffer_vector_transform",
    ],
)

py_library(
    name = "pack_host_buffer_vector_module",
    srcs = ["pack_host_buffer_vector_module.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":pack_host_buffer_vector_transform_pybind",
        "@xla_serving//data/processor/python:composite_processor_pybind",
        "@xla_serving//data/processor/python:transform",
    ],
)

py_library(
    name = "unpack_host_buffer_vector_module",
    srcs = ["unpack_host_buffer_vector_module.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":unpack_host_buffer_vector_transform_pybind",
        "@xla_serving//data/processor/python:transform",
        "@xla_serving_pypi//jax",
    ],
)
