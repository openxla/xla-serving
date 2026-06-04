load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = [
    "@neptune//:__subpackages__",
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
        "@neptune//protos:manifest_cc_proto",
        "@neptune//runtime/core:ifrt_loaded_variable_registry",
        "@neptune//runtime/interpreter:manifest_utils",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/platform:env",
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
        "@neptune//protos:manifest_cc_proto",
        "@neptune//runtime/core:constants",
        "@neptune//runtime/core:ifrt_device_utils",
        "@neptune//runtime/core:ifrt_loaded_variable_registry",
        "@neptune//runtime/core:stablehlo_utils",
        "@neptune//runtime/core:value",
        "@neptune//runtime/data_loader/ocp_loader:checkpoint_loader",
        "@neptune//runtime/data_loader/ocp_loader:metadata_util",
        "@neptune//runtime/interpreter:manifest_utils",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/ir:hlo_sharding",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/concurrency:future",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
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
        "@neptune//protos:manifest_cc_proto",
    ],
    alwayslink = 1,
)
