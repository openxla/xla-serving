load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    visibility = ["@neptune//runtime/serving/gcp:__subpackages__"],
    deps = [
        "@com_github_grpc_grpc//:grpc++",
        "@com_github_grpc_grpc//:grpc_security_base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
    ],
)
