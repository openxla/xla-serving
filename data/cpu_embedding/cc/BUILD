load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = [
    "@neptune//:__subpackages__",
    "@neptune//data/cpu_embedding:__subpackages__",
])

cc_library(
    name = "compiled_custom_combiner",
    srcs = ["compiled_custom_combiner.cc"],
    hdrs = ["compiled_custom_combiner.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@eigen_archive//:eigen3",
        "@neptune//data/kernels/cpu_embedding:cpu_embedding_cc_proto",
        "@xla//xla/backends/cpu/nanort:nanort_client",
        "@xla//xla/backends/cpu/nanort:nanort_executable",
        "@xla//xla/hlo/builder:xla_computation",
        "@xla//xla/pjrt:mlir_to_hlo",
        "@xla//xla/tsl/concurrency:async_value",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_test(
    name = "compiled_custom_combiner_test",
    srcs = ["compiled_custom_combiner_test.cc"],
    defines = [
        "ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS",
        "ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS",
    ],
    deps = [
        ":compiled_custom_combiner",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
        "@neptune//data/kernels/cpu_embedding:cpu_embedding_cc_proto",
        "@neptune//data/test_utils:status_macros",
    ],
)

cc_library(
    name = "embedding_table",
    srcs = ["embedding_table.cc"],
    hdrs = ["embedding_table.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        ":array",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "cpu_embedding_engine",
    srcs = ["cpu_embedding_engine.cc"],
    hdrs = ["cpu_embedding_engine.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        ":array",
        ":compute_activations",
        ":embedding_table",
        ":feature_batch_csr_input",
        ":feature_spec",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/functional:any_invocable",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/types:span",
        "@eigen_archive//:eigen3",
        "@xla//xla/tsl/platform:status_macros",
    ],
)

cc_library(
    name = "feature_batch_csr_input",
    srcs = ["feature_batch_csr_input.cc"],
    hdrs = ["feature_batch_csr_input.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@xla//xla/tsl/platform:status_macros",
    ],
)

cc_library(
    name = "compute_activations",
    srcs = ["compute_activations.cc"],
    hdrs = ["compute_activations.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        ":array",
        ":embedding_table",
        ":feature_batch_csr_input",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
        "@eigen_archive//:eigen3",
        "@xla//xla/tsl/platform:status_macros",
    ],
)

cc_test(
    name = "embedding_table_test",
    srcs = ["embedding_table_test.cc"],
    defines = [
        "ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS",
        "ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS",
    ],
    deps = [
        ":array",
        ":embedding_table",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
        "@neptune//data/test_utils:status_macros",
    ],
)

cc_test(
    name = "compute_activations_test",
    srcs = ["compute_activations_test.cc"],
    defines = [
        "ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS",
        "ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS",
    ],
    deps = [
        ":array",
        ":compute_activations",
        ":embedding_table",
        ":feature_batch_csr_input",
        "@abseil-cpp//absl/types:span",
        "@eigen_archive//:eigen3",
        "@googletest//:gtest_main",
        "@neptune//data/test_utils:status_macros",
    ],
)

cc_test(
    name = "feature_batch_csr_input_test",
    srcs = ["feature_batch_csr_input_test.cc"],
    defines = [
        "ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS",
        "ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS",
    ],
    deps = [
        ":feature_batch_csr_input",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@googletest//:gtest_main",
        "@neptune//data/test_utils:status_macros",
    ],
)

cc_test(
    name = "cpu_embedding_engine_test",
    srcs = ["cpu_embedding_engine_test.cc"],
    defines = [
        "ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS",
        "ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS",
    ],
    deps = [
        ":array",
        ":cpu_embedding_engine",
        ":embedding_table",
        ":feature_batch_csr_input",
        ":feature_spec",
        "@abseil-cpp//absl/types:span",
        "@eigen_archive//:eigen3",
        "@googletest//:gtest_main",
        "@neptune//data/test_utils:status_macros",
    ],
)

cc_library(
    name = "feature_spec",
    hdrs = ["feature_spec.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = ["@abseil-cpp//absl/strings:str_format"],
)

cc_library(
    name = "array",
    hdrs = ["array.h"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS"],
    deps = [
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/types:span",
        "@eigen_archive//:eigen3",
    ],
)

cc_test(
    name = "array_test",
    srcs = ["array_test.cc"],
    defines = [
        "ABSL_DEFINE_UNQUALIFIED_STATUS_MACROS",
        "ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS",
    ],
    deps = [
        ":array",
        "@abseil-cpp//absl/types:span",
        "@eigen_archive//:eigen3",
        "@googletest//:gtest_main",
    ],
)
