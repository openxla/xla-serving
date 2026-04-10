load("@rules_license//rules:license.bzl", "license")
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//devtools/python/blaze:strict.bzl", "py_library_providing_imports_info")

licenses(["notice"])

license(
    name = "license",
    package_name = "neptune-model",
)

py_library_providing_imports_info(
    name = "neptune_model",
    srcs = ["__init__.py"],
    lazy_imports = True,
    lib_rule = pytype_strict_library,
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/neptune/model/_src/export:jax_model",
        "//third_party/neptune/model/_src/export:main_lib",
        "//third_party/neptune/model/_src/export:serving_config",
    ],
)
