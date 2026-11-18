load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@xla_serving//data/processor/python:build_defs.bzl", "transform_pybind_library")
load("@xla_serving//data/resource_manager/python:build_defs.bzl", "resource_pybind_library")

package(default_visibility = ["@xla_serving//data:xla_serving_data_internal"])

cc_library(
    name = "add_int_transform",
    testonly = 1,
    srcs = ["add_int_transform.cc"],
    hdrs = ["add_int_transform.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@xla_serving//data/processor/cc/api:registry",
        "@xla_serving//data/processor/cc/api:shape",
        "@xla_serving//data/processor/cc/api:transform_context",
        "@xla_serving//data/processor/cc/api:type_info",
        "@xla_serving//data/processor/cc/core:transform",
        "@xla_serving//data/processor/proto:transform_config_cc_proto",
        "@xla_serving//data/resource_manager/cc:resource",
    ],
    alwayslink = 1,
)

transform_pybind_library(
    name = "add_int_transform_pybind",
    testonly = 1,
    srcs = ["add_int_transform.cc"],
    hdrs = ["add_int_transform.h"],
    dynamic_deps = ["@xla_serving//data:xla_serving_data_test_core"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../../../../xla_serving_data_test:$$ORIGIN/../../../../xla_serving_data"],
    deps = [
        ":add_int_transform",
    ],
)

cc_library(
    name = "add_host_buffer_transform",
    testonly = 1,
    srcs = ["add_host_buffer_transform.cc"],
    hdrs = ["add_host_buffer_transform.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@xla//xla/python/ifrt",
        "@xla_serving//data/processor/cc/api:registry",
        "@xla_serving//data/processor/cc/core:transform",
        "@xla_serving//data/processor/proto:transform_config_cc_proto",
        "@xla_serving//data/resource_manager/cc:resource",
        "@xla_serving//runtime/core:host_buffer",
    ],
    alwayslink = 1,
)

transform_pybind_library(
    name = "add_host_buffer_transform_pybind",
    testonly = 1,
    srcs = ["add_host_buffer_transform.cc"],
    hdrs = ["add_host_buffer_transform.h"],
    dynamic_deps = ["@xla_serving//data:xla_serving_data_test_core"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../../../../xla_serving_data_test:$$ORIGIN/../../../../xla_serving_data"],
    deps = [":add_host_buffer_transform"],
)

cc_library(
    name = "add_two_transform",
    testonly = 1,
    srcs = ["add_two_transform.cc"],
    hdrs = ["add_two_transform.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@xla_serving//data/processor/cc/api:registry",
        "@xla_serving//data/processor/cc/api:transform_context",
        "@xla_serving//data/processor/cc/api:type_info",
        "@xla_serving//data/processor/cc/core:transform",
        "@xla_serving//data/processor/proto:transform_config_cc_proto",
        "@xla_serving//data/resource_manager/cc:resource",
    ],
    alwayslink = 1,
)

transform_pybind_library(
    name = "add_two_transform_pybind",
    testonly = 1,
    srcs = ["add_two_transform.cc"],
    hdrs = ["add_two_transform.h"],
    dynamic_deps = ["@xla_serving//data:xla_serving_data_test_core"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../../../../xla_serving_data_test:$$ORIGIN/../../../../xla_serving_data"],
    deps = [":add_two_transform"],
)

cc_library(
    name = "multiply_by_five_transform",
    testonly = 1,
    srcs = ["multiply_by_five_transform.cc"],
    hdrs = ["multiply_by_five_transform.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@xla_serving//data/processor/cc/api:registry",
        "@xla_serving//data/processor/cc/api:transform_context",
        "@xla_serving//data/processor/cc/api:type_info",
        "@xla_serving//data/processor/cc/core:transform",
        "@xla_serving//data/processor/proto:transform_config_cc_proto",
        "@xla_serving//data/resource_manager/cc:resource",
    ],
    alwayslink = 1,
)

transform_pybind_library(
    name = "multiply_by_five_transform_pybind",
    testonly = 1,
    srcs = ["multiply_by_five_transform.cc"],
    hdrs = ["multiply_by_five_transform.h"],
    dynamic_deps = ["@xla_serving//data:xla_serving_data_test_core"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../../../../xla_serving_data_test:$$ORIGIN/../../../../xla_serving_data"],
    deps = [":multiply_by_five_transform"],
)

cc_library(
    name = "dummy_resource_transform",
    testonly = 1,
    srcs = ["dummy_resource_transform.cc"],
    hdrs = ["dummy_resource_transform.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/types:span",
        "@xla_serving//data/processor/cc/api:registry",
        "@xla_serving//data/processor/cc/core:transform",
        "@xla_serving//data/processor/proto:transform_config_cc_proto",
        "@xla_serving//data/resource_manager/cc:asset",
        "@xla_serving//data/resource_manager/cc:resource",
        "@xla_serving//data/resource_manager/proto:resource_manager_cc_proto",
    ],
    alwayslink = 1,
)

transform_pybind_library(
    name = "dummy_resource_transform_pybind",
    testonly = 1,
    srcs = ["dummy_resource_transform.cc"],
    hdrs = ["dummy_resource_transform.h"],
    dynamic_deps = ["@xla_serving//data:xla_serving_data_test_core"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../../../../xla_serving_data_test:$$ORIGIN/../../../../xla_serving_data"],
    deps = [":dummy_resource_transform"],
)

cc_library(
    name = "suffix_transform",
    testonly = 1,
    srcs = ["suffix_transform.cc"],
    hdrs = ["suffix_transform.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@com_google_protobuf//:wrappers_cc_proto",
        "@xla_serving//data/processor/cc/api:registry",
        "@xla_serving//data/processor/cc/api:transform_context",
        "@xla_serving//data/processor/cc/api:type_info",
        "@xla_serving//data/processor/cc/core:transform",
        "@xla_serving//data/processor/proto:transform_config_cc_proto",
    ],
    alwayslink = 1,
)

transform_pybind_library(
    name = "suffix_transform_pybind",
    testonly = 1,
    srcs = ["suffix_transform.cc"],
    hdrs = ["suffix_transform.h"],
    dynamic_deps = ["@xla_serving//data:xla_serving_data_test_core"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../../../../xla_serving_data_test:$$ORIGIN/../../../../xla_serving_data"],
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
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        ":int_resource",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@xla_serving//data/processor/cc/api:registry",
        "@xla_serving//data/processor/cc/core:transform",
        "@xla_serving//data/processor/proto:transform_config_cc_proto",
        "@xla_serving//data/resource_manager/cc:resource",
    ],
    alwayslink = 1,
)

transform_pybind_library(
    name = "add_from_resource_transform_pybind",
    testonly = 1,
    srcs = ["add_from_resource_transform.cc"],
    hdrs = ["add_from_resource_transform.h"],
    dynamic_deps = ["@xla_serving//data:xla_serving_data_test_core"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../../../../xla_serving_data_test:$$ORIGIN/../../../../xla_serving_data"],
    deps = [":add_from_resource_transform"],
)

cc_library(
    name = "int_resource",
    testonly = 1,
    srcs = ["int_resource.cc"],
    hdrs = ["int_resource.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@xla_serving//data/resource_manager/cc:resource",
    ],
    alwayslink = 1,
)

resource_pybind_library(
    name = "int_resource_pybind",
    testonly = 1,
    srcs = ["int_resource.cc"],
    hdrs = ["int_resource.h"],
    dynamic_deps = ["@xla_serving//data:xla_serving_data_test_core"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../../../../xla_serving_data_test:$$ORIGIN/../../../../xla_serving_data"],
    deps = [":int_resource"],
)

cc_library(
    name = "make_dict_transform",
    testonly = 1,
    srcs = ["make_dict_transform.cc"],
    hdrs = ["make_dict_transform.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@xla_serving//data/processor/cc/api:registry",
        "@xla_serving//data/processor/cc/api:transform_context",
        "@xla_serving//data/processor/cc/api:type_info",
        "@xla_serving//data/processor/cc/core:transform",
        "@xla_serving//data/processor/proto:transform_config_cc_proto",
        "@xla_serving//data/resource_manager/cc:resource",
    ],
    alwayslink = 1,
)

transform_pybind_library(
    name = "make_dict_transform_pybind",
    testonly = 1,
    srcs = ["make_dict_transform.cc"],
    hdrs = ["make_dict_transform.h"],
    dynamic_deps = ["@xla_serving//data:xla_serving_data_test_core"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../../../../xla_serving_data_test:$$ORIGIN/../../../../xla_serving_data"],
    deps = [":make_dict_transform"],
)

cc_library(
    name = "multi_kernel_math_transform",
    testonly = 1,
    srcs = ["multi_kernel_math_transform.cc"],
    hdrs = ["multi_kernel_math_transform.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@xla//xla/python/ifrt",
        "@xla_serving//data/processor/cc/api:registry",
        "@xla_serving//data/processor/cc/api:transform_context",
        "@xla_serving//data/processor/cc/core:transform",
        "@xla_serving//data/processor/proto:transform_config_cc_proto",
        "@xla_serving//data/resource_manager/cc:resource",
        "@xla_serving//runtime/core:host_buffer",
        "@xla_serving//runtime/core:host_buffer_type_utils",
    ],
    alwayslink = 1,
)

transform_pybind_library(
    name = "multi_kernel_math_transform_pybind",
    testonly = 1,
    srcs = ["multi_kernel_math_transform.cc"],
    hdrs = ["multi_kernel_math_transform.h"],
    dynamic_deps = ["@xla_serving//data:xla_serving_data_test_core"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../../../../xla_serving_data_test:$$ORIGIN/../../../../xla_serving_data"],
    deps = [":multi_kernel_math_transform"],
)
