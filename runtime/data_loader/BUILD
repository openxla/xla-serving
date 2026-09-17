load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = [
    "@xla_serving//:__subpackages__",
])

package_group(
    name = "checkpoint_users",
    packages = [
        "//runtime/interpreter/...",
        "//smartass/brain/lemming/...",
    ],
)

cc_library(
    name = "data_loader",
    srcs = ["data_loader.cc"],
    hdrs = ["data_loader.h"],
    visibility = [":checkpoint_users"],
    deps = [
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/functional:any_invocable",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/types:span",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/platform:env",
        "@xla_serving//protos:manifest_cc_proto",
        "@xla_serving//runtime/core:ifrt_loaded_variable_registry",
        "@xla_serving//runtime/interpreter:manifest_utils",
    ],
)

cc_library(
    name = "default_data_loader",
    srcs = ["default_data_loader.cc"],
    hdrs = ["default_data_loader.h"],
    deps = [
        ":data_loader",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/ir:hlo_sharding",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/concurrency:future",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
        "@xla_serving//protos:manifest_cc_proto",
        "@xla_serving//runtime/core:constants",
        "@xla_serving//runtime/core:ifrt_device_utils",
        "@xla_serving//runtime/core:ifrt_loaded_variable_registry",
        "@xla_serving//runtime/core:stablehlo_utils",
        "@xla_serving//runtime/core:value",
        "@xla_serving//runtime/data_loader/ocp_loader:checkpoint_loader",
        "@xla_serving//runtime/data_loader/ocp_loader:metadata_util",
        "@xla_serving//runtime/interpreter:manifest_utils",
    ],
)

cc_library(
    name = "default_data_loader_registration",
    srcs = ["default_data_loader_registration.cc"],
    visibility = [":checkpoint_users"],
    deps = [
        ":data_loader",
        ":default_data_loader",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status:statusor",
        "@xla_serving//protos:manifest_cc_proto",
    ],
    alwayslink = 1,
)
