"""Build definitions for transform registry pybind."""

load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

def transform_pybind_library(name, srcs = [], hdrs = [], deps = [], use_pytype = True, generate_wrappers = False, **kwargs):
    """Generates a pybind extension for a list of Transform subclasses.

    Args:
      name: The name of the rule.
      srcs: List of C++ source files containing the ND_REGISTER_TRANSFORM macros.
      hdrs: List of header files containing the Transform subclasses.
      deps: List of dependencies.
      use_pytype: Whether to use pytype_pybind_extension (and generate a .pyi python stub file).
      generate_wrappers: Whether to auto-generate a `_module.py` providing Transform python wrappers and op functions.
      **kwargs: Additional keyword arguments passed to pybind_extension and genrule.
    """
    cc_file = name + "_gen.cc"
    all_inputs = srcs + hdrs

    # We use a python script to parse the source files and generate the pybind C++ code.
    # This avoids relying on C++ macros to register pybind types, which is fragile
    # and requires wrapper headers.
    native.genrule(
        name = name + "_gen_cc",
        srcs = all_inputs,
        outs = [cc_file],
        tools = ["@xla_serving//data/processor/python:generate_pybind"],
        cmd = "$(location @xla_serving//data/processor/python:generate_pybind) $(location " + cc_file + ") " + name + " $(SRCS)",
        testonly = kwargs.get("testonly", False),
    )

    if use_pytype:
        pyi_file = name + ".pyi"
        native.genrule(
            name = name + "_gen_pyi",
            srcs = all_inputs,
            outs = [pyi_file],
            tools = ["@xla_serving//data/processor/python:generate_pybind"],
            cmd = "$(location @xla_serving//data/processor/python:generate_pybind) $(location " + pyi_file + ") " + name + " $(SRCS)",
            testonly = kwargs.get("testonly", False),
        )

        pybind_extension(
            name = name,
            srcs = [cc_file],
            
            
            deps = deps + [
                "@xla_serving//data/processor/cc/core:transform",
                "@xla_serving//data/processor/python:transform_pybind",
                "@pybind11_abseil//pybind11_abseil:status_casters",
                "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
                "@xla_serving//data/processor/proto:transform_config_cc_proto",
            ],
            
            
            **kwargs
        )
    else:
        pybind_extension(
            name = name,
            srcs = [cc_file],
            
            deps = deps + [
                "@xla_serving//data/processor/cc/core:transform",
                "@xla_serving//data/processor/python:transform_pybind",
                "@pybind11_abseil//pybind11_abseil:status_casters",
                "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
                "@xla_serving//data/processor/proto:transform_config_cc_proto",
            ],
            
            **kwargs
        )
