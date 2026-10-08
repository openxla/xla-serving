load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["@xla_serving//:__subpackages__"])

exports_files(glob(["*.h"]))

cc_library(
    name = "op_functors",
    hdrs = ["op_functors.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
)
