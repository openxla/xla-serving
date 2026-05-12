load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "model_inspector",
    srcs = ["model_inspector.cc"],
    hdrs = ["model_inspector.h"],
    deps = [
        "//file/base",
        "//file/base:path",
        "//third_party/gloop/util/status:status_macros",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@neptune//protos:manifest_cc_proto",
        "@neptune//protos:orchestration_cc_proto",
    ],
)
