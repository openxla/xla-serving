load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = [
    "@xla_serving//:__subpackages__",
])

cc_library(
    name = "tensorstore_utils_cc",
    srcs = ["tensorstore_utils.cc"],
    hdrs = ["tensorstore_utils.h"],
    deps = [
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
        "@nlohmann_json//:json",
        "@tensorstore//tensorstore:data_type",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "metadata_util",
    srcs = ["metadata_util.cc"],
    hdrs = ["metadata_util.h"],
    deps = [
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@nlohmann_json//:json",
        "@tsl//tsl/platform:path",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_test(
    name = "tensorstore_utils_test_cc",
    srcs = ["tensorstore_utils_test.cc"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS"],
    deps = [
        ":tensorstore_utils_cc",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
        "@tensorstore//tensorstore:data_type",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/ifrt:test_util",
        "@xla//xla/python/pjrt_ifrt:tfrt_cpu_client_test_lib",
        "@xla//xla/tsl/concurrency:ref_count",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_test(
    name = "metadata_util_test",
    srcs = ["metadata_util_test.cc"],
    data = [
        "testdata/checkpoint_with_ocp_metadata/_METADATA",
        "testdata/checkpoint_with_ocp_metadata/_METADATA_SKIP_DESERIALIZE",
    ],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS"],
    deps = [
        ":metadata_util",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
        "@tsl//tsl/platform:path",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:resource_loader",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "checkpoint_loader",
    srcs = ["checkpoint_loader.cc"],
    hdrs = ["checkpoint_loader.h"],
    deps = [
        ":common_types",
        ":tensorstore_loader",
        ":tensorstore_utils_cc",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@nlohmann_json//:json",
        "@tensorstore//tensorstore/driver/zarr",  # tensorstore driver
        "@tensorstore//tensorstore/kvstore/ocdbt",
        "@tsl//tsl/platform:path",
        "@xla//xla:shape_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt:pjrt_dtype",
        "@xla//xla/python/pjrt_ifrt:xla_ifrt",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_test(
    name = "checkpoint_loader_test",
    srcs = ["checkpoint_loader_test.cc"],
    data = [
        "@xla_serving//runtime/data_loader/testdata:non_spmd_checkpoint",
        "@xla_serving//runtime/data_loader/testdata:non_spmd_ocdbt",
        "@xla_serving//runtime/data_loader/testdata:spmd_checkpoint",
        "@xla_serving//runtime/data_loader/testdata:spmd_ocdbt",
    ],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS"],
    deps = [
        ":checkpoint_loader",
        ":common_types",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/ifrt:test_util",
        "@xla//xla/python/pjrt_ifrt:tfrt_cpu_client_test_lib",
        "@xla//xla/python/pjrt_ifrt:xla_ifrt",
        "@xla//xla/tsl/concurrency:future",
        "@xla//xla/tsl/concurrency:ref_count",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:resource_loader",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:test",
        "@xla_serving//protos:manifest_cc_proto",
        "@xla_serving//runtime/core:host_buffer",
        "@xla_serving//runtime/core:host_buffer_matcher",
        "@xla_serving//runtime/core:host_buffer_test_utils",
        "@xla_serving//runtime/core:sharding_utils",
    ],
)

cc_library(
    name = "tensorstore_loader",
    srcs = ["tensorstore_loader.cc"],
    hdrs = ["tensorstore_loader.h"],
    deps = [
        ":common_types",
        ":shared",
        ":tensorstore_utils_cc",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/cleanup",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@abseil-cpp//absl/types:span",
        "@com_google_protobuf//:protobuf_lite",
        "@nlohmann_json//:json",
        "@tensorstore//tensorstore",
        "@tensorstore//tensorstore:array",
        "@tensorstore//tensorstore:context",
        "@tensorstore//tensorstore:contiguous_layout",
        "@tensorstore//tensorstore:data_type",
        "@tensorstore//tensorstore:index",
        "@tensorstore//tensorstore:open",
        "@tensorstore//tensorstore:open_mode",
        "@tensorstore//tensorstore/index_space:dim_expression",
        "@tensorstore//tensorstore/index_space:index_transform",
        "@tensorstore//tensorstore/util:element_pointer",
        "@tensorstore//tensorstore/util:result",
        "@tsl//tsl/platform:path",
        "@xla//xla:shape_util",
        "@xla//xla:util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/pjrt:pjrt_layout",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/ifrt:with_user_context",
        "@xla//xla/python/pjrt_ifrt:pjrt_dtype",
        "@xla//xla/python/pjrt_ifrt:pjrt_layout",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_test(
    name = "tensorstore_loader_test",
    srcs = ["tensorstore_loader_test.cc"],
    data = [
        "@xla_serving//runtime/data_loader/testdata:non_spmd_checkpoint",
        "@xla_serving//runtime/data_loader/testdata:spmd_ocdbt",
    ],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS"],
    deps = [
        ":common_types",
        ":shared",
        ":tensorstore_loader",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
        "@tensorstore//tensorstore/driver/zarr",  # tensorstore driver
        "@tensorstore//tensorstore/kvstore/file",
        "@tensorstore//tensorstore/kvstore/ocdbt",
        "@tsl//tsl/platform:path",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/ifrt:test_util",
        "@xla//xla/python/pjrt_ifrt:tfrt_cpu_client_test_lib",
        "@xla//xla/python/pjrt_ifrt:xla_ifrt",
        "@xla//xla/tsl/concurrency:future",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:resource_loader",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:test",
        "@xla_serving//runtime/core:host_buffer",
        "@xla_serving//runtime/core:host_buffer_matcher",
        "@xla_serving//runtime/core:host_buffer_test_utils",
        "@xla_serving//runtime/core:sharding_utils",
    ],
)

cc_library(
    name = "shared",
    hdrs = ["shared.h"],
    deps = [
        "@abseil-cpp//absl/status:statusor",
        "@nlohmann_json//:json",
        "@tensorstore//tensorstore:context",
        "@tensorstore//tensorstore/kvstore/file",  # for OSS
        "@tensorstore//tensorstore/util:result",
    ],
)

cc_library(
    name = "common_types",
    hdrs = ["common_types.h"],
)
