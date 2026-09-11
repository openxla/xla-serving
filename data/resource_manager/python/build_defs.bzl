"""Build definitions for resource manager pybind."""

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

def resource_pybind_library(name, srcs = [], hdrs = [], deps = [], **kwargs):
    """Generates a pybind extension for a list of Resource subclasses.

    Args:
      name: The name of the rule.
      srcs: List of C++ source files containing the ND_REGISTER_RESOURCE macros.
      hdrs: List of header files containing the Resource subclasses.
      deps: List of dependencies.
      **kwargs: Additional keyword arguments passed to pybind_extension.
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
        tools = ["@xla_serving//data/resource_manager/python:generate_pybind"],
        cmd = "$(location @xla_serving//data/resource_manager/python:generate_pybind) $(location " + cc_file + ") " + name + " $(SRCS)",
    )

    pybind_extension(
        name = name,
        srcs = [cc_file],
        
        deps = deps + [
            "@xla_serving//data/resource_manager/cc:resource",
            "@xla_serving//data/resource_manager/python:resource_pybind_lib",
            "@pybind11_abseil//pybind11_abseil:status_casters",
            "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
            "@xla_serving//data/resource_manager/proto:resource_manager_cc_proto",
        ],
        
        **kwargs
    )
