load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = [
    "@neptune//:__subpackages__",
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
        "@xla//xla/hlo/ir:hlo_sharding",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt:xla_ifrt",
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
