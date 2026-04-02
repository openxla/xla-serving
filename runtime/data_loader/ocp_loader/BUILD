load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = [
    "//learning/infra/mira/experimental/orbax_model:__subpackages__",
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
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/python/ifrt",
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
        "//third_party/australis:cpu_support",  # backend
        "//third_party/australis/google/ifrt:ifrt_australis_cc_proto",
        "@com_google_protobuf//:protobuf",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@nlohmann_json//:json",
        "@tensorstore//tensorstore/driver/zarr",  # tensorstore driver
        "@tensorstore//tensorstore/_only:all_kvstore_drivers",  # tensorstore
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
        "//third_party/australis/google/ifrt:ifrt_australis_cc_proto",
        "//third_party/australis/google/ifrt:tensorstore_loader",
        "@com_google_protobuf//:protobuf_lite",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/time",
        "@nlohmann_json//:json",
        "@tensorstore//tensorstore:context",
        "@tensorstore//tensorstore/util:result",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "shared",
    hdrs = ["shared.h"],
    deps = [
        "@abseil-cpp//absl/status:statusor",
        "@tensorstore//tensorstore:context",
        "@tensorstore//tensorstore/util:result",
    ],
)

cc_library(
    name = "common_types",
    hdrs = ["common_types.h"],
)
