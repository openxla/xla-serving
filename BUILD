# Package: neptune
#
# Description:
#   Neptune is a serving engine for serving generic ML workloads on GPUs and TPUs.
#   See the README.md file for more details.

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_license//rules:license.bzl", "license")

package(default_visibility = [":internal"])

license(
    name = "license",
    package_name = "neptune",
)

licenses(["notice"])

# Packages that have access to neptune-internal implementation details.
package_group(
    name = "internal",
    packages = [
        "//...",
    ],
)

exports_files([
    "LICENSE",
])
