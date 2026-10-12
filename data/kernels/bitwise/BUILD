load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["@xla_serving//:__subpackages__"])

exports_files(glob([
    "op_*.cc",
    "*.h",
]))

filegroup(
    name = "bitwise_ops_srcs",
    srcs = glob(["op_*.cc"]),
)

cc_library(
    name = "op_functors",
    hdrs = ["op_functors.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
)
