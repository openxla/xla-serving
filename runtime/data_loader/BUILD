load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = [
    "@neptune//:__subpackages__",
])

package_group(
    name = "checkpoint_users",
    packages = [
        "@neptune//runtime/interpreter/...",
    ],
)

cc_library(
    name = "data_loader",
    srcs = ["data_loader.cc"],
    hdrs = ["data_loader.h"],
    visibility = [":checkpoint_users"],
    deps = [
        "//third_party/py/orbax/experimental/model/core/protos:manifest_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
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
        "//third_party/py/orbax/experimental/model/core/protos:manifest_cc_proto",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
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
        "//third_party/py/orbax/experimental/model/core/protos:manifest_cc_proto",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = 1,
)
