load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = [
    "//learning/infra/mira/experimental/orbax_model:__subpackages__",
    "@neptune//:__subpackages__",
])

cc_library(
    name = "serving_core_selector",
    srcs = ["serving_core_selector.cc"],
    hdrs = ["serving_core_selector.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@xla//xla/tsl/framework:serving_device_selector",
        "@xla//xla/tsl/lib/monitoring:counter",
    ],
)
