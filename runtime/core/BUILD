load("//devtools/deps/check:deps_check.bzl", "check_dependencies")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

# Limit visibility to selected packages.
package_group(
    name = "core_users",
    packages = [
        "//learning/brain/experimental/jax_data/...",
        "//learning/infra/mira/experimental/orbax_model/core/...",
        "//learning/infra/mira/experimental/orbax_model/interpreter/...",
        "//learning/infra/mira/experimental/orbax_model/python/...",
        "//learning/infra/mira/experimental/orbax_model/tensorflow/...",
        "//smartass/brain/server/...",
        "@neptune//runtime/core/google/...",
        "@neptune//runtime/data_loader/...",
        "@neptune//runtime/interpreter/...",
        "@neptune//runtime/kernel/...",
        "@neptune//runtime/python/...",
        "@neptune//runtime/serving/...",
    ],
)

filegroup(
    name = "testdata",
    srcs = glob([
        "testdata/*.mlir",
    ]),
    visibility = ["@neptune//runtime/core/google:__subpackages__"],
)

cc_library(
    name = "constants",
    hdrs = ["constants.h"],
    visibility = [
        ":core_users",
        "//learning/serving/controller/validation:__subpackages__",
    ],
)

cc_library(
    name = "common_util",
    srcs = ["common_util.cc"],
    hdrs = ["common_util.h"],
    visibility = [
        "//learning/infra/mira/experimental/orbax_model/core:__pkg__",
        "//learning/serving/servolab:__subpackages__",
        "//learning/tfx/pipeline/infra_validation:__pkg__",
        "//smartass/brain/util:__pkg__",
    ],
    deps = [
        ":constants",
        "//file/base:path",
        "//file/colossus/public:cns",
        "//file/localfile",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
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
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
    ],
)

cc_library(
    name = "type",
    hdrs = ["type.h"],
    visibility = [":core_users"],
    deps = [
        ":value",
        "@com_google_absl//absl/container:btree",
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
        "//testing/base/public:gunit_for_library_testonly",
    ],
)

cc_library(
    name = "sharding_utils_core",
    srcs = ["sharding_utils_core.cc"],
    hdrs = ["sharding_utils_core.h"],
    visibility = [":core_users"],
    deps = [
        ":host_buffer",
        "//third_party/eigen3",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
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
        "//third_party/eigen3",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@xla//xla:shape_util",
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
    name = "stablehlo_utils",
    srcs = ["stablehlo_utils.cc"],
    hdrs = ["stablehlo_utils.h"],
    visibility = [":core_users"],
    deps = [
        "//third_party/openxla/shardy/src/shardy/dialect/sdy/ir:dialect",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@neptune//runtime/libs/mlir:utils",
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
    name = "stablehlo_runner",
    srcs = ["stablehlo_runner.cc"],
    hdrs = ["stablehlo_runner.h"],
    visibility = [":core_users"],
    deps = [
        ":host_buffer",
        ":ifrt_device_utils",
        ":ifrt_loaded_variable_registry",
        ":loaded_stablehlo",
        ":sharding_utils",
        ":stablehlo",
        ":stablehlo_optimizer",
        ":stablehlo_utils",
        ":value",
        "//third_party/protobuf",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@neptune//runtime/interpreter:manifest_utils",
        "@neptune//runtime/libs/ifrt:executable_build_utils",
        "@neptune//runtime/libs/ifrt/persistent_compilation_cache:ifrt_persistent_compilation_cache",
        "@neptune//runtime/libs/ifrt/serving_core_selector",
        "@neptune//runtime/libs/mlir:dump_mlir_util",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//xla:shape_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/ir:hlo_sharding",
        "@xla//xla/pjrt/proto:compile_options_proto_cc",
        "@xla//xla/python:refine_polymorphic_shapes",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt:xla_ifrt",
        "@xla//xla/service:computation_placer_hdr",
        "@xla//xla/tsl/concurrency:future",
        "@xla//xla/tsl/concurrency:ref_count",
        "@xla//xla/tsl/framework:serving_device_selector",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "stablehlo_runner_test_lib",
    testonly = True,
    srcs = ["stablehlo_runner_test.cc"],
    data = [
        ":testdata",
    ],
    visibility = ["@neptune//runtime/core/google:__subpackages__"],
    deps = [
        ":host_buffer",
        ":host_buffer_matcher",
        ":host_buffer_test_utils",
        ":ifrt_loaded_variable_registry",
        ":sharding_utils",
        ":stablehlo",
        ":stablehlo_runner",
        ":stablehlo_utils",
        ":value",
        ":value_utils",
        "//devtools/build/runtime:get_runfiles_dir",
        "//testing/base/public:gunit_main",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@neptune//runtime/interpreter:manifest_utils",
        "@neptune//runtime/libs/ifrt/serving_core_selector",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/ir:hlo_sharding",
        "@xla//xla/hlo/ir:tile_assignment",
        "@xla//xla/pjrt/proto:compile_options_proto_cc",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/ifrt:test_util",
        "@xla//xla/python/pjrt_ifrt:xla_ifrt",
        "@xla//xla/tsl/concurrency:future",
        "@xla//xla/tsl/framework:serving_device_selector",
        "@xla//xla/tsl/framework/test_util:mock_serving_device_selector",
        "@xla//xla/tsl/lib/core:status_test_util",
        "@xla//xla/tsl/lib/monitoring:cell_reader",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "host_buffer",
    srcs = ["host_buffer.cc"],
    hdrs = ["host_buffer.h"],
    visibility = [
        ":core_users",
        "@neptune//runtime/interpreter:interpreter_users",
    ],
    deps = [
        ":allocator",
        ":default_allocator",
        ":host_buffer_type_utils",
        "//third_party/australis/google/ifrt:host_array",
        "//third_party/eigen3",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@neptune//runtime/core/proto:host_buffer_cc_proto",
        "@tsl//tsl/platform:tstring",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/concurrency:ref_count",
        "@xla//xla/tsl/framework:allocator",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:types",
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
        "//testing/base/public:gunit_for_library_testonly",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/types:span",
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
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
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
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
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
        "//learning/infra/mira/experimental/orbax_model/core:__subpackages__",
        "//learning/infra/mira/experimental/orbax_model/tensorflow:__subpackages__",
        "@neptune//runtime/interpreter:__subpackages__",
        "@neptune//runtime/interpreter:interpreter_users",
        "@neptune//runtime/kernel:__subpackages__",
    ],
    deps = [
        ":stablehlo_utils",
        ":value",
        "//third_party/py/orbax/experimental/model/core/protos:manifest_cc_proto",
        "//third_party/py/orbax/experimental/model/core/protos:type_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "grid",
    srcs = ["grid.cc"],
    hdrs = ["grid.h"],
    deps = [
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "ifrt_device_utils",
    srcs = ["ifrt_device_utils.cc"],
    hdrs = ["ifrt_device_utils.h"],
    visibility = [":core_users"],
    deps = [
        ":grid",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
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
    deps = [
        ":stablehlo",
        ":stablehlo_utils",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
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
    name = "pjrt_ifrt_cpu_client_test_lib",
    testonly = True,
    srcs = ["pjrt_ifrt_cpu_client_test_lib.cc"],
    visibility = [":core_users"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@xla//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "@xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/ifrt:test_util",
        "@xla//xla/python/pjrt_ifrt",
        "@xla//xla/tsl/platform:statusor",
    ],
    alwayslink = True,
)

cc_library(
    name = "pjrt_c_api_ifrt_cpu_client_test_lib",
    testonly = True,
    srcs = ["pjrt_c_api_ifrt_cpu_client_test_lib.cc"],
    visibility = [":core_users"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt/c_api_client:pjrt_c_api_client",
        "@xla//xla/pjrt/plugin:plugin_names",
        "@xla//xla/pjrt/plugin/xla_cpu:cpu_static_registration",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/ifrt:test_util",
        "@xla//xla/python/pjrt_ifrt",
        "@xla//xla/tsl/platform:statusor",
    ],
    alwayslink = True,
)

cc_library(
    name = "nanort_cpu_client_test_lib",
    testonly = True,
    srcs = ["nanort_cpu_client_test_lib.cc"],
    visibility = ["//visibility:private"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@xla//xla/backends/cpu/nanort:ifrt_client",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/ifrt:test_util",
    ],
    alwayslink = True,
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
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "incremental_barrier",
    srcs = ["incremental_barrier.cc"],
    hdrs = ["incremental_barrier.h"],
    visibility = [":core_users"],
    deps = [
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log:check",
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
        "//third_party/protobuf",
        "//third_party/py/orbax/experimental/model/core/protos:manifest_cc_proto",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
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
        "@com_google_absl//absl/status",
        "@llvm-project//mlir:IR",
        "@xla//xla/pjrt/proto:compile_options_proto_cc",
    ],
)

cc_library(
    name = "allocator",
    hdrs = ["allocator.h"],
    visibility = [
        ":core_users",
        "//learning/brain/tfrt/orbax_model:__subpackages__",
        "//learning/serving/servables/tfrt:__subpackages__",
        "@neptune//runtime/core/batching:__subpackages__",
        "@neptune//runtime/core/google:__subpackages__",
    ],
)

cc_library(
    name = "default_allocator",
    hdrs = ["default_allocator.h"],
    visibility = [
        ":core_users",
        "//learning/brain/tfrt/orbax_model:__subpackages__",
        "@neptune//runtime/core/batching:__subpackages__",
    ],
    deps = [
        ":allocator",
        "@com_google_absl//absl/base:no_destructor",
    ],
)

cc_library(
    name = "thread_pool_interface",
    hdrs = ["thread_pool_interface.h"],
    visibility = [
        ":core_users",
        "//learning/infra/mira/experimental/orbax_model/tensorflow:__subpackages__",
    ],
    deps = [
        "//third_party/eigen3",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@xla//xla/tsl/platform:env",
    ],
)

cc_library(
    name = "common_thread_pool",
    hdrs = ["common_thread_pool.h"],
    deps = [
        ":thread_pool_interface",
        "@com_google_absl//absl/status",
        "@xla//xla/tsl/platform:env",
    ],
)

check_dependencies(
    name = "stablehlo_runner_deps_test",
    of = [":stablehlo_runner"],
    allowed_match_regexp = "|".join([
        # go/keep-sorted start
        "//third_party/tensorflow/compiler/tf2xla(/|:)",  # brought in by "@stablehlo//:stablehlo_serialization"
        "//third_party/tensorflow/compiler/xla(/|:)",  # OpenXLA
        "//third_party/tensorflow/core/.*proto",  # brought in by "//platforms/xla/service/jellyfish:tpu_compilation_environment_cc_proto"
        "//third_party/tensorflow/core/grappler/costs:op_performance_data",  # brought in by "@xla//xla/service/gpu:model_config_store"
        "//third_party/tensorflow/core/platform(/|:)",  # contains data types. Brought in by several libs.
        "//third_party/tensorflow/core:protos_all",  # brought in by "@xla//xla:xla_data_proto_cc"
        "//third_party/tensorflow/third_party(/|:)",
        "//third_party/tensorflow/tsl(/|:)",  # go/moving-tsl-into-xla-lsc
        "//third_party/tensorflow:",  # contains platform-related libs. Brought in by several libs.
        # go/keep-sorted end
    ]),
    dont_match_regexp = "//third_party/tensorflow",
    error_context = "'StableHloRunner' should not have any dependency on TensorFlow.",
)
