load("@rules_license//rules:license.bzl", "license")
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")

licenses(["notice"])

license(
    name = "license",
    package_name = "neptune-model",
)

py_library(
    name = "neptune_model",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@neptune//neptune_model/_src/export:jax_model",
        "@neptune//neptune_model/_src/export:main_lib",
        "@neptune//neptune_model/_src/export:serving_config",
        "@neptune//neptune_model/_src/processors:processor",
        "@neptune//neptune_model/_src/processors:tf_processor",
    ],
)
