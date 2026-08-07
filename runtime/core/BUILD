load("@rules_cc//cc:cc_library.bzl", "cc_library")

# Limit visibility to selected packages.
package_group(
    name = "core_users",
    packages = [
        "//runtime/core/google/...",
        "//runtime/core/stablehlo_transforms/...",
        "//runtime/data_loader/...",
        "//runtime/interpreter/...",
        "//runtime/kernel/...",
        "//runtime/python/...",
        "//runtime/serving/...",
        "//runtime/tensorflow/...",
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
        "@neptune//runtime/interpreter:interpreter_users",
    ],
    deps = [
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
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
        ":host_buffer",
        ":host_buffer_type_utils",
        ":sharding_utils_core",
        "@abseil-cpp//absl/container:btree",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@eigen_archive//:eigen3",
        "@llvm-project//llvm:Support",
        "@tsl//tsl/profiler/lib:traceme",
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
    name = "sharding_utils_test_lib",
    testonly = True,
    srcs = ["sharding_utils_test.cc"],
    visibility = ["@neptune//runtime/core/google:__subpackages__"],
    deps = [
        ":host_buffer",
        ":host_buffer_matcher",
        ":host_buffer_test_utils",
        ":sharding_utils",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@googletest//:gtest_main",
        "@tsl//tsl/platform:ml_dtypes",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/ir:hlo",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/ifrt:test_util",
        "@xla//xla/python/pjrt_ifrt:xla_ifrt",
        "@xla//xla/tsl/concurrency:ref_count",
        "@xla//xla/tsl/lib/core:status_test_util",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:statusor",
    ],
    alwayslink = 1,
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
        "@neptune//runtime/libs/mlir:utils",
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
    ],
)

cc_library(
    name = "device_buffer",
    hdrs = ["device_buffer.h"],
    visibility = [":core_users"],
    deps = ["@xla//xla/python/ifrt"],
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
        "@com_google_protobuf//:protobuf",
        "@abseil-cpp//absl/base",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
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
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@neptune//runtime/core/streaming",
        "@neptune//runtime/core/streaming:utils",
        "@neptune//runtime/interpreter:context",
        "@neptune//runtime/interpreter:manifest_utils",
        "@neptune//runtime/libs/ifrt:executable_build_utils",
        "@neptune//runtime/libs/ifrt/persistent_compilation_cache:ifrt_persistent_compilation_cache",
        "@neptune//runtime/libs/ifrt/serving_core_selector",
        "@neptune//runtime/libs/mlir:dump_mlir_util",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//xla:shape_util",
        "@xla//xla:status_macros",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/ir:hlo_sharding",
        "@xla//xla/pjrt/proto:compile_options_proto_cc",
        "@xla//xla/python:refine_polymorphic_shapes",
        "@xla//xla/python/ifrt",
        "@xla//xla/service:computation_placer_hdr",
        "@xla//xla/tsl/concurrency:future",
        "@xla//xla/tsl/concurrency:ref_count",
        "@xla//xla/tsl/framework:serving_device_selector",
        "@xla//xla/tsl/lib/monitoring:counter",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "host_buffer",
    srcs = ["host_buffer.cc"],
    hdrs = ["host_buffer.h"],
    visibility = [
        ":core_users",
        "@neptune//examples:__subpackages__",
        "@neptune//runtime/interpreter:interpreter_users",
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
        "@neptune//protos/grpc:host_buffer_cc_proto",
        "@tsl//tsl/platform:tstring",
        "@xla//xla:util",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/framework:allocator",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:types",
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
        "@neptune//runtime/interpreter:__subpackages__",
        "@neptune//runtime/interpreter:interpreter_users",
        "@neptune//runtime/kernel:__subpackages__",
        "@neptune//runtime/tensorflow:__subpackages__",
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
        "@neptune//protos:manifest_cc_proto",
        "@neptune//protos:type_cc_proto",
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
        "@xla//xla/service:computation_placer_hdr",
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
        "@neptune//runtime/libs/ifrt:stablehlo_loader",
        "@neptune//runtime/libs/mlir:dump_mlir_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/pjrt/proto:compile_options_proto_cc",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "host_buffer_type_utils",
    hdrs = ["host_buffer_type_utils.h"],
    deps = [
        "@tsl//tsl/platform:bfloat16",
        "@tsl//tsl/platform:ml_dtypes",
        "@tsl//tsl/platform:tstring",
        "@xla//xla/python/ifrt",
    ],
)

cc_library(
    name = "value_utils",
    hdrs = ["value_utils.h"],
    visibility = [":core_users"],
    deps = [
        ":value",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
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
        "@com_google_protobuf//:protobuf",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@neptune//protos:manifest_cc_proto",
        "@neptune//runtime/interpreter:manifest_utils",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/tsl/platform:statusor",
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
        "@neptune//runtime/core/batching:__subpackages__",
        "@neptune//runtime/core/google:__subpackages__",
    ],
)

cc_library(
    name = "default_allocator",
    hdrs = ["default_allocator.h"],
    visibility = [
        ":core_users",
        "@neptune//runtime/core/batching:__subpackages__",
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
        "@neptune//runtime/tensorflow:__subpackages__",
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
