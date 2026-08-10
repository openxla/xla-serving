load("@neptune//data/processor/python:build_defs.bzl", "transform_pybind_library")
load("@neptune//data/resource_manager/python:build_defs.bzl", "resource_pybind_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["@neptune//data:neptune_data_internal"])

cc_library(
    name = "add_int_transform",
    testonly = 1,
    srcs = ["add_int_transform.cc"],
    hdrs = ["add_int_transform.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@neptune//data/processor/cc/api:registry",
        "@neptune//data/processor/cc/api:transform_context",
        "@neptune//data/processor/cc/api:type_info",
        "@neptune//data/processor/cc/core:transform",
        "@neptune//data/processor/proto:transform_config_cc_proto",
        "@neptune//data/resource_manager/cc:resource",
    ],
    alwayslink = 1,
)

transform_pybind_library(
    name = "add_int_transform_pybind",
    testonly = 1,
    srcs = ["add_int_transform.cc"],
    hdrs = ["add_int_transform.h"],
    dynamic_deps = ["@neptune//data:neptune_data_test_core"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../../../../neptune_data_test:$$ORIGIN/../../../../neptune_data"],
    deps = [
        ":add_int_transform",
    ],
)

cc_library(
    name = "add_host_buffer_transform",
    testonly = 1,
    srcs = ["add_host_buffer_transform.cc"],
    hdrs = ["add_host_buffer_transform.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@neptune//data/processor/cc/api:registry",
        "@neptune//data/processor/cc/core:transform",
        "@neptune//data/processor/proto:transform_config_cc_proto",
        "@neptune//data/resource_manager/cc:resource",
        "@neptune//runtime/core:host_buffer",
        "@xla//xla/python/ifrt",
    ],
    alwayslink = 1,
)

transform_pybind_library(
    name = "add_host_buffer_transform_pybind",
    testonly = 1,
    srcs = ["add_host_buffer_transform.cc"],
    hdrs = ["add_host_buffer_transform.h"],
    dynamic_deps = ["@neptune//data:neptune_data_test_core"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../../../../neptune_data_test:$$ORIGIN/../../../../neptune_data"],
    deps = [":add_host_buffer_transform"],
)

cc_library(
    name = "add_two_transform",
    testonly = 1,
    srcs = ["add_two_transform.cc"],
    hdrs = ["add_two_transform.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@neptune//data/processor/cc/api:registry",
        "@neptune//data/processor/cc/api:transform_context",
        "@neptune//data/processor/cc/api:type_info",
        "@neptune//data/processor/cc/core:transform",
        "@neptune//data/processor/proto:transform_config_cc_proto",
        "@neptune//data/resource_manager/cc:resource",
    ],
    alwayslink = 1,
)

transform_pybind_library(
    name = "add_two_transform_pybind",
    testonly = 1,
    srcs = ["add_two_transform.cc"],
    hdrs = ["add_two_transform.h"],
    dynamic_deps = ["@neptune//data:neptune_data_test_core"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../../../../neptune_data_test:$$ORIGIN/../../../../neptune_data"],
    deps = [":add_two_transform"],
)

cc_library(
    name = "multiply_by_five_transform",
    testonly = 1,
    srcs = ["multiply_by_five_transform.cc"],
    hdrs = ["multiply_by_five_transform.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@neptune//data/processor/cc/api:registry",
        "@neptune//data/processor/cc/api:transform_context",
        "@neptune//data/processor/cc/api:type_info",
        "@neptune//data/processor/cc/core:transform",
        "@neptune//data/processor/proto:transform_config_cc_proto",
        "@neptune//data/resource_manager/cc:resource",
    ],
    alwayslink = 1,
)

transform_pybind_library(
    name = "multiply_by_five_transform_pybind",
    testonly = 1,
    srcs = ["multiply_by_five_transform.cc"],
    hdrs = ["multiply_by_five_transform.h"],
    dynamic_deps = ["@neptune//data:neptune_data_test_core"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../../../../neptune_data_test:$$ORIGIN/../../../../neptune_data"],
    deps = [":multiply_by_five_transform"],
)

cc_library(
    name = "dummy_resource_transform",
    testonly = 1,
    srcs = ["dummy_resource_transform.cc"],
    hdrs = ["dummy_resource_transform.h"],
    deps = [
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/types:span",
        "@neptune//data/processor/cc/api:registry",
        "@neptune//data/processor/cc/core:transform",
        "@neptune//data/processor/proto:transform_config_cc_proto",
        "@neptune//data/resource_manager/cc:asset",
        "@neptune//data/resource_manager/cc:resource",
        "@neptune//data/resource_manager/proto:resource_manager_cc_proto",
    ],
    alwayslink = 1,
)

transform_pybind_library(
    name = "dummy_resource_transform_pybind",
    testonly = 1,
    srcs = ["dummy_resource_transform.cc"],
    hdrs = ["dummy_resource_transform.h"],
    dynamic_deps = ["@neptune//data:neptune_data_test_core"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../../../../neptune_data_test:$$ORIGIN/../../../../neptune_data"],
    deps = [":dummy_resource_transform"],
)

cc_library(
    name = "suffix_transform",
    testonly = 1,
    srcs = ["suffix_transform.cc"],
    hdrs = ["suffix_transform.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@com_google_protobuf//:wrappers_cc_proto",
        "@neptune//data/processor/cc/api:registry",
        "@neptune//data/processor/cc/api:transform_context",
        "@neptune//data/processor/cc/api:type_info",
        "@neptune//data/processor/cc/core:transform",
        "@neptune//data/processor/proto:transform_config_cc_proto",
        "@neptune//data/resource_manager/cc:resource",
    ],
    alwayslink = 1,
)

transform_pybind_library(
    name = "suffix_transform_pybind",
    testonly = 1,
    srcs = ["suffix_transform.cc"],
    hdrs = ["suffix_transform.h"],
    dynamic_deps = ["@neptune//data:neptune_data_test_core"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../../../../neptune_data_test:$$ORIGIN/../../../../neptune_data"],
    deps = [":suffix_transform"],
)

cc_library(
    name = "add_from_resource_transform",
    testonly = 1,
    srcs = ["add_from_resource_transform.cc"],
    hdrs = [
        "add_from_resource_transform.h",
        "int_resource.h",
    ],
    deps = [
        ":int_resource",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@neptune//data/processor/cc/api:registry",
        "@neptune//data/processor/cc/core:transform",
        "@neptune//data/processor/proto:transform_config_cc_proto",
        "@neptune//data/resource_manager/cc:resource",
    ],
    alwayslink = 1,
)

transform_pybind_library(
    name = "add_from_resource_transform_pybind",
    testonly = 1,
    srcs = ["add_from_resource_transform.cc"],
    hdrs = ["add_from_resource_transform.h"],
    dynamic_deps = ["@neptune//data:neptune_data_test_core"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../../../../neptune_data_test:$$ORIGIN/../../../../neptune_data"],
    deps = [":add_from_resource_transform"],
)

cc_library(
    name = "int_resource",
    testonly = 1,
    srcs = ["int_resource.cc"],
    hdrs = ["int_resource.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@neptune//data/resource_manager/cc:resource",
    ],
    alwayslink = 1,
)

resource_pybind_library(
    name = "int_resource_pybind",
    testonly = 1,
    srcs = ["int_resource.cc"],
    hdrs = ["int_resource.h"],
    dynamic_deps = ["@neptune//data:neptune_data_test_core"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../../../../neptune_data_test:$$ORIGIN/../../../../neptune_data"],
    deps = [":int_resource"],
)

cc_library(
    name = "make_dict_transform",
    testonly = 1,
    srcs = ["make_dict_transform.cc"],
    hdrs = ["make_dict_transform.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@neptune//data/processor/cc/api:registry",
        "@neptune//data/processor/cc/api:transform_context",
        "@neptune//data/processor/cc/api:type_info",
        "@neptune//data/processor/cc/core:transform",
        "@neptune//data/processor/proto:transform_config_cc_proto",
        "@neptune//data/resource_manager/cc:resource",
    ],
    alwayslink = 1,
)

transform_pybind_library(
    name = "make_dict_transform_pybind",
    testonly = 1,
    srcs = ["make_dict_transform.cc"],
    hdrs = ["make_dict_transform.h"],
    dynamic_deps = ["@neptune//data:neptune_data_test_core"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../../../../neptune_data_test:$$ORIGIN/../../../../neptune_data"],
    deps = [":make_dict_transform"],
)
