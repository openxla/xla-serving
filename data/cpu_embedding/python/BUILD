load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")
load("@rules_python//python:py_test.bzl", "py_test")

package(default_visibility = ["@neptune//data/cpu_embedding:__subpackages__"])

pybind_library(
    name = "array_caster",
    hdrs = ["array_caster.h"],
    visibility = [
        "@neptune//:__subpackages__",
        "@neptune//data/cpu_embedding:__subpackages__",
    ],
    deps = [
        "@abseil-cpp//absl/status",
        "@neptune//data/cpu_embedding/cc:array",
    ],
)

pybind_extension(
    name = "cpu_embedding_engine_pybind",
    srcs = [
        "cpu_embedding_engine_pybind.cc",
    ],
    visibility = [
        "@neptune//:__subpackages__",
        "@neptune//data/cpu_embedding:__subpackages__",
    ],
    deps = [
        ":array_caster",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@neptune//data/cpu_embedding/cc:array",
        "@neptune//data/cpu_embedding/cc:cpu_embedding_engine",
        "@neptune//data/cpu_embedding/cc:embedding_table",
        "@neptune//data/cpu_embedding/cc:feature_batch_csr_input",
        "@neptune//data/cpu_embedding/cc:feature_spec",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:status_casters",
    ],
)

py_test(
    name = "cpu_embedding_engine_test",
    srcs = ["cpu_embedding_engine_test.py"],
    deps = [
        ":cpu_embedding_engine_pybind",
        "@absl_py//absl/testing:absltest",
        "@neptune_pypi//numpy",
    ],
)
