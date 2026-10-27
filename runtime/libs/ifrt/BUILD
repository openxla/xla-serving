load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = [
    "@xla_serving//:__subpackages__",
])

filegroup(
    name = "testdata",
    srcs = glob([
        "testdata/*.mlir",
    ]),
)

genrule(
    name = "serialized_stablehlo",
    srcs = [
        ":testdata",
    ],
    outs = [file.replace(".mlir", ".serializedmlir") for file in glob(["testdata/*.mlir"])],
    cmd = """
        for FILE in $(SRCS); do
          out=$(@D)/$$(basename $${FILE} .mlir).serializedmlir
          $(execpath @stablehlo//:stablehlo-translate) --serialize --target=0.9.0 $$FILE > $$out
        done
    """,
    tools = [
        "@stablehlo//:stablehlo-translate",
    ],
)

cc_library(
    name = "stablehlo_loader",
    srcs = ["stablehlo_loader.cc"],
    hdrs = ["stablehlo_loader.h"],
    deps = [
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:TensorDialect",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:register",
        "@stablehlo//:stablehlo_ops",
        "@xla//xla/pjrt:mlir_to_hlo",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_test(
    name = "stablehlo_loader_test",
    srcs = ["stablehlo_loader_test.cc"],
    data = [
        ":serialized_stablehlo",
        ":testdata",
    ],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS"],
    deps = [
        ":stablehlo_loader",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:resource_loader",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "executable_build_utils",
    srcs = ["executable_build_utils.cc"],
    hdrs = ["executable_build_utils.h"],
    defines = ["XLA_SERVING_ON_GCE"],
    deps = [
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/time",
        "@llvm-project//mlir:IR",
        "@xla//xla:debug_options_flags",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/pjrt:pjrt_compiler",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt/proto:compile_options_proto_cc",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/ifrt/hlo:hlo_program",
        "@xla//xla/python/pjrt_ifrt:xla_ifrt",
        "@xla//xla/service:compilation_environments",
        "@xla//xla/service:device_assignment",
        "@xla//xla/tsl/platform:statusor",
        "@xla_serving//runtime/libs/ifrt/persistent_compilation_cache:ifrt_persistent_compilation_cache",
    ],
)

cc_test(
    name = "executable_build_utils_test",
    srcs = ["executable_build_utils_test.cc"],
    data = [
        ":testdata",
    ],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS"],
    deps = [
        ":executable_build_utils",
        ":stablehlo_loader",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/functional:any_invocable",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@abseil-cpp//absl/types:span",
        "@com_google_protobuf//:any_cc_proto",
        "@com_google_protobuf//:wrappers_cc_proto",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@xla//xla/backends/cpu/nanort:ifrt_client",
        "@xla//xla/pjrt:pjrt_common",
        "@xla//xla/pjrt:pjrt_compiler",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt/proto:compile_options_proto_cc",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/ifrt:mock",
        "@xla//xla/python/ifrt:ref_wrapper",
        "@xla//xla/python/ifrt:test_util",
        "@xla//xla/python/ifrt/hlo:hlo_program",
        "@xla//xla/python/pjrt_ifrt:tfrt_cpu_client_test_lib",
        "@xla//xla/python/pjrt_ifrt:xla_ifrt",
        "@xla//xla/service:compilation_environments",
        "@xla//xla/service:device_assignment",
        "@xla//xla/tsl/concurrency:ref_count",
        "@xla//xla/tsl/lib/core:status_test_util",
        "@xla//xla/tsl/platform:resource_loader",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/util/proto:proto_matchers",
        "@xla_serving//runtime/libs/ifrt/persistent_compilation_cache:ifrt_persistent_compilation_cache",
    ],
)

cc_library(
    name = "executable_build_utils_gce",
    srcs = ["executable_build_utils.cc"],
    hdrs = ["executable_build_utils.h"],
    defines = ["XLA_SERVING_ON_GCE"],
    visibility = [
        "//visibility:private",
    ],
    deps = [
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/time",
        "@llvm-project//mlir:IR",
        "@xla//xla:debug_options_flags",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/pjrt:pjrt_compiler",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt/proto:compile_options_proto_cc",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/ifrt/hlo:hlo_program",
        "@xla//xla/python/pjrt_ifrt:xla_ifrt",
        "@xla//xla/service:compilation_environments",
        "@xla//xla/service:device_assignment",
        "@xla//xla/tsl/platform:statusor",
        "@xla_serving//runtime/libs/ifrt/persistent_compilation_cache:ifrt_persistent_compilation_cache",
    ],
)

cc_test(
    name = "executable_build_utils_gce_test",
    srcs = ["executable_build_utils_gce_test.cc"],
    data = [
        ":testdata",
    ],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS"],
    deps = [
        ":executable_build_utils_gce",
        ":stablehlo_loader",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@com_google_protobuf//:any_cc_proto",
        "@com_google_protobuf//:protobuf",
        "@com_google_protobuf//:wrappers_cc_proto",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt/proto:compile_options_proto_cc",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/ifrt:mock",
        "@xla//xla/python/ifrt:ref_wrapper",
        "@xla//xla/python/pjrt_ifrt:tfrt_cpu_client_test_lib",
        "@xla//xla/python/pjrt_ifrt:xla_ifrt",
        "@xla//xla/service:compilation_environments",
        "@xla//xla/service:device_assignment",
        "@xla//xla/tsl/concurrency:ref_count",
        "@xla//xla/tsl/platform:resource_loader",
    ],
)
