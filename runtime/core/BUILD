load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

# Limit visibility to selected packages.
package_group(
    name = "core_users",
    packages = [
        "//...",
    ],
)

cc_library(
    name = "constants",
    hdrs = ["constants.h"],
    visibility = [
        ":core_users",
    ],
)

cc_library(
    name = "common_util",
    srcs = ["common_util.cc"],
    hdrs = ["common_util.h"],
    visibility = [
        ":core_users",
    ],
    deps = [
        ":constants",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:string_view",
        "@tsl//tsl/platform:path",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
    ],
)

cc_library(
    name = "value",
    hdrs = ["value.h"],
    visibility = [
        ":core_users",
        "@xla_serving//runtime/interpreter:interpreter_users",
    ],
    deps = [
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
    ],
)

cc_test(
    name = "value_test",
    srcs = ["value_test.cc"],
    deps = [
        ":value",
        "@abseil-cpp//absl/strings:string_view",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "type",
    hdrs = ["type.h"],
    visibility = [":core_users"],
    deps = [
        ":value",
        "@abseil-cpp//absl/container:btree",
    ],
)

cc_library(
    name = "value_matcher",
    testonly = True,
    srcs = [],
    hdrs = ["value_matcher.h"],
    visibility = [":core_users"],
    deps = [
        ":value",
        "@googletest//:gtest",
    ],
)

cc_test(
    name = "value_matcher_test",
    srcs = ["value_matcher_test.cc"],
    deps = [
        ":value",
        ":value_matcher",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "sharding_utils_core",
    srcs = ["sharding_utils_core.cc"],
    hdrs = ["sharding_utils_core.h"],
    visibility = [":core_users"],
    deps = [
        ":host_buffer",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@eigen_archive//:eigen3",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:macros",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "sharding_utils",
    srcs = [
        "sharding_utils.cc",
    ],
    hdrs = [
        "sharding_utils.h",
    ],
    visibility = [":core_users"],
    deps = [
        ":device_buffer",
        ":host_buffer",
        ":host_buffer_type_utils",
        ":host_buffer_utils",
        ":sharding_utils_core",
        "@abseil-cpp//absl/container:btree",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@eigen_archive//:eigen3",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//xla:status_macros",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt:xla_ifrt",
        "@xla//xla/tsl/concurrency:future",
        "@xla//xla/tsl/concurrency:ref_count",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "h2d_transfer_executor",
    hdrs = ["h2d_transfer_executor.h"],
    visibility = [":core_users"],
    deps = [
        ":host_buffer",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/platform:env",
    ],
)

cc_library(
    name = "stablehlo_utils",
    srcs = ["stablehlo_utils.cc"],
    hdrs = ["stablehlo_utils.h"],
    visibility = [":core_users"],
    deps = [
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_passes",
        "@xla//xla:shape_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/hlo/parser:hlo_parser",
        "@xla//xla/mlir/utils:error_util",
        "@xla//xla/mlir/utils:type_util",
        "@xla//xla/pjrt/proto:compile_options_proto_cc",
        "@xla//xla/python:refine_polymorphic_shapes",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt:pjrt_dtype",
        "@xla//xla/service/spmd/shardy:utils",
        "@xla//xla/service/spmd/shardy/stablehlo_round_trip:export_shardings",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
        "@xla_serving//runtime/libs/mlir:utils",
    ],
)

cc_library(
    name = "device_buffer",
    hdrs = ["device_buffer.h"],
    visibility = [":core_users"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "stablehlo_runner",
    srcs = [
        "aot_stablehlo_runner.cc",
        "stablehlo_runner.cc",
    ],
    hdrs = [
        "aot_stablehlo_runner.h",
        "stablehlo_runner.h",
    ],
    visibility = [":core_users"],
    deps = [
        ":device_buffer",
        ":h2d_transfer_executor",
        ":host_buffer",
        ":ifrt_device_utils",
        ":ifrt_loaded_variable_registry",
        ":loaded_stablehlo",
        ":sharding_utils",
        ":stablehlo",
        ":stablehlo_optimizer",
        ":stablehlo_utils",
        ":value",
        ":value_utils",
        "@abseil-cpp//absl/base",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/hash",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/log:vlog_is_on",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@abseil-cpp//absl/types:span",
        "@com_google_protobuf//:protobuf",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//xla:shape_util",
        "@xla//xla:status_macros",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/ir:hlo_sharding",
        "@xla//xla/pjrt/proto:compile_options_proto_cc",
        "@xla//xla/python:refine_polymorphic_shapes",
        "@xla//xla/python/ifrt",
        "@xla//xla/service:computation_placer",
        "@xla//xla/tsl/concurrency:future",
        "@xla//xla/tsl/concurrency:ref_count",
        "@xla//xla/tsl/framework:serving_device_selector",
        "@xla//xla/tsl/lib/monitoring:counter",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
        "@xla_serving//runtime/core/streaming",
        "@xla_serving//runtime/core/streaming:utils",
        "@xla_serving//runtime/interpreter:context",
        "@xla_serving//runtime/interpreter:manifest_utils",
        "@xla_serving//runtime/libs/ifrt:executable_build_utils",
        "@xla_serving//runtime/libs/ifrt/persistent_compilation_cache:ifrt_persistent_compilation_cache",
        "@xla_serving//runtime/libs/ifrt/serving_core_selector",
        "@xla_serving//runtime/libs/mlir:dump_mlir_util",
    ],
)

cc_library(
    name = "host_buffer",
    srcs = ["host_buffer.cc"],
    hdrs = ["host_buffer.h"],
    visibility = [
        ":core_users",
        "@xla_serving//examples:__subpackages__",
        "@xla_serving//runtime/interpreter:interpreter_users",
    ],
    deps = [
        ":allocator",
        ":default_allocator",
        ":host_buffer_type_utils",
        ":host_buffer_utils",
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:span",
        "@eigen_archive//:eigen3",
        "@tsl//tsl/platform:tstring",
        "@xla//xla:util",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/framework:allocator",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:types",
        "@xla_serving//protos/grpc:host_buffer_cc_proto",
    ],
)

cc_library(
    name = "host_buffer_utils",
    srcs = ["host_buffer_utils.cc"],
    hdrs = ["host_buffer_utils.h"],
    deps = [
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@tsl//tsl/platform:bfloat16",
        "@tsl//tsl/platform:tstring",
        "@xla//xla:util",
        "@xla//xla/python/ifrt",
    ],
)

cc_library(
    name = "host_buffer_matcher",
    testonly = True,
    srcs = ["host_buffer_matcher.cc"],
    hdrs = ["host_buffer_matcher.h"],
    visibility = [":core_users"],
    deps = [
        ":host_buffer",
        ":host_buffer_type_utils",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest",
        "@xla//xla/python/ifrt",
    ],
)

cc_library(
    name = "host_buffer_test_utils",
    testonly = True,
    hdrs = ["host_buffer_test_utils.h"],
    visibility = [":core_users"],
    deps = [
        ":allocator",
        ":default_allocator",
        ":host_buffer",
        ":host_buffer_type_utils",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/types:span",
        "@xla//xla/python/ifrt",
    ],
)

cc_library(
    name = "ifrt_loaded_variable_registry",
    srcs = ["ifrt_loaded_variable_registry.cc"],
    hdrs = ["ifrt_loaded_variable_registry.h"],
    visibility = [":core_users"],
    deps = [
        ":value",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/functional:any_invocable",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@xla//xla:shape_util",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "flat_signature",
    srcs = ["flat_signature.cc"],
    hdrs = ["flat_signature.h"],
    visibility = [
        ":core_users",
        "@xla_serving//runtime/interpreter:__subpackages__",
        "@xla_serving//runtime/interpreter:interpreter_users",
        "@xla_serving//runtime/kernel:__subpackages__",
        "@xla_serving//runtime/tensorflow:__subpackages__",
    ],
    deps = [
        ":stablehlo_utils",
        ":value",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "grid",
    srcs = ["grid.cc"],
    hdrs = ["grid.h"],
    deps = [
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
    ],
)

cc_library(
    name = "ifrt_device_utils",
    srcs = ["ifrt_device_utils.cc"],
    hdrs = ["ifrt_device_utils.h"],
    visibility = [":core_users"],
    deps = [
        ":grid",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/ifrt:attribute_map",
        "@xla//xla/service:computation_placer",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "loaded_stablehlo",
    srcs = ["loaded_stablehlo.cc"],
    hdrs = ["loaded_stablehlo.h"],
    visibility = [":core_users"],
    deps = [
        ":stablehlo",
        ":stablehlo_utils",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:vlog_is_on",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/types:span",
        "@llvm-project//mlir:IR",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/pjrt/proto:compile_options_proto_cc",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
        "@xla_serving//runtime/libs/ifrt:stablehlo_loader",
        "@xla_serving//runtime/libs/mlir:dump_mlir_util",
    ],
)

cc_library(
    name = "host_buffer_type_utils",
    hdrs = ["host_buffer_type_utils.h"],
    visibility = [":core_users"],
    deps = [
        "@tsl//tsl/platform:bfloat16",
        "@tsl//tsl/platform:ml_dtypes",
        "@tsl//tsl/platform:tstring",
        "@xla//xla/python/ifrt",
    ],
)

cc_library(
    name = "value_utils",
    srcs = ["value_utils.cc"],
    hdrs = ["value_utils.h"],
    visibility = [":core_users"],
    deps = [
        ":device_buffer",
        ":host_buffer",
        ":value",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "incremental_barrier",
    srcs = ["incremental_barrier.cc"],
    hdrs = ["incremental_barrier.h"],
    visibility = [":core_users"],
    deps = [
        "@abseil-cpp//absl/functional:any_invocable",
        "@abseil-cpp//absl/functional:bind_front",
        "@abseil-cpp//absl/log:check",
    ],
)

cc_library(
    name = "stablehlo",
    srcs = ["stablehlo.cc"],
    hdrs = ["stablehlo.h"],
    visibility = [
        ":core_users",
    ],
    deps = [
        ":stablehlo_utils",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@com_google_protobuf//:protobuf",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/tsl/platform:statusor",
        "@xla_serving//protos:manifest_cc_proto",
        "@xla_serving//runtime/interpreter:manifest_utils",
    ],
)

cc_library(
    name = "stablehlo_optimizer",
    srcs = ["stablehlo_optimizer.cc"],
    hdrs = ["stablehlo_optimizer.h"],
    visibility = [":core_users"],
    deps = [
        "@abseil-cpp//absl/status",
        "@llvm-project//mlir:IR",
        "@xla//xla/pjrt/proto:compile_options_proto_cc",
    ],
)

cc_library(
    name = "allocator",
    hdrs = ["allocator.h"],
    visibility = [
        ":core_users",
        "@xla_serving//runtime/core/batching:__subpackages__",
        "@xla_serving//runtime/core/google:__subpackages__",
    ],
)

cc_library(
    name = "default_allocator",
    hdrs = ["default_allocator.h"],
    visibility = [
        ":core_users",
        "@xla_serving//runtime/core/batching:__subpackages__",
    ],
    deps = [
        ":allocator",
        "@abseil-cpp//absl/base:no_destructor",
    ],
)

cc_library(
    name = "thread_pool_interface",
    hdrs = ["thread_pool_interface.h"],
    visibility = [
        ":core_users",
        "@xla_serving//runtime/tensorflow:__subpackages__",
    ],
    deps = [
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/status",
        "@eigen_archive//:eigen3",
        "@xla//xla/tsl/platform:env",
    ],
)

cc_library(
    name = "common_thread_pool",
    hdrs = ["common_thread_pool.h"],
    deps = [
        ":thread_pool_interface",
        "@abseil-cpp//absl/status",
        "@xla//xla/tsl/platform:env",
    ],
)

cc_library(
    name = "batched_h2d_transfer_executor",
    srcs = ["batched_h2d_transfer_executor.cc"],
    hdrs = ["batched_h2d_transfer_executor.h"],
    visibility = ["@xla_serving//runtime:__subpackages__"],
    deps = [
        ":h2d_transfer_executor",
        ":host_buffer",
        ":host_buffer_utils",
        ":sharding_utils",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@xla//xla:literal",
        "@xla//xla:shape_util",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_layout",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt",
        "@xla//xla/python/pjrt_ifrt:pjrt_dtype",
        "@xla//xla/python/pjrt_ifrt:xla_ifrt",
        "@xla//xla/tsl/concurrency:ref_count",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:statusor",
    ],
)
