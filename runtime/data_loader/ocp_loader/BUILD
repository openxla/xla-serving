load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = [
    "//learning/infra/mira/experimental/orbax_model:__subpackages__",
    "@neptune//:__subpackages__",
])

cc_library(
    name = "tensorstore_utils_cc",
    srcs = ["tensorstore_utils.cc"],
    hdrs = ["tensorstore_utils.h"],
    deps = [
        "//third_party/json",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "metadata_util",
    srcs = ["metadata_util.cc"],
    hdrs = ["metadata_util.h"],
    deps = [
        "//file/base:path",
        "//third_party/json",
        "//util/task:status",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
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
        "//file/base:path",
        "//third_party/australis:cpu_support",  # backend
        "//third_party/australis/google/ifrt:ifrt_australis_cc_proto",
        "//third_party/json",
        "//third_party/protobuf",
        "//third_party/tensorstore/driver/zarr",  # tensorstore driver
        "//third_party/tensorstore/google3_only:all_kvstore_drivers",  # tensorstore
        "//third_party/tensorstore/kvstore/ocdbt",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
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
        "//third_party/json",
        "@com_google_protobuf//:protobuf_lite",
        "//third_party/tensorstore:context",
        "//third_party/tensorstore/util:result",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "shared",
    hdrs = ["shared.h"],
    deps = [
        "//third_party/tensorstore:context",
        "//third_party/tensorstore/util:result",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "common_types",
    hdrs = ["common_types.h"],
)
