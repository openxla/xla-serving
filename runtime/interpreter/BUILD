load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package_group(
    name = "internal",
    packages = [
        "//runtime/interpreter/...",
        "//runtime/serving/...",
    ],
)

package_group(
    name = "interpreter_users",
    packages = [
        "//examples/...",
        "//runtime/...",
    ],
)

package(default_visibility = [
    ":internal",
])

cc_library(
    name = "optional_function_handler",
    hdrs = ["optional_function_handler.h"],
    visibility = [
        "@xla_serving//runtime/tensorflow:__subpackages__",
        "@xla_serving//runtime/xla_serving_data:__subpackages__",
    ],
    deps = [
        ":interpreter_create_options",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@xla_serving//protos:manifest_cc_proto",
        "@xla_serving//protos:orchestration_cc_proto",
        "@xla_serving//runtime/kernel",
        "@xla_serving//runtime/kernel:kernel_io_data_type",
    ],
)

cc_library(
    name = "optional_function_handler_factory",
    srcs = ["optional_function_handler_factory.cc"],
    hdrs = ["optional_function_handler_factory.h"],
    visibility = [
        "@xla_serving//runtime/tensorflow:__subpackages__",
        "@xla_serving//runtime/xla_serving_data:__subpackages__",
    ],
    deps = [
        ":optional_function_handler",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/synchronization",
    ],
)

cc_library(
    name = "orbax_type_utils",
    srcs = ["orbax_type_utils.cc"],
    hdrs = ["orbax_type_utils.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/python/ifrt",
        "@xla_serving//protos:type_cc_proto",
    ],
)

cc_library(
    name = "manifest_utils",
    srcs = ["manifest_utils.cc"],
    hdrs = ["manifest_utils.h"],
    visibility = [
        "@xla_serving//runtime/core:__subpackages__",
        "@xla_serving//runtime/data_loader:__subpackages__",
        "@xla_serving//runtime/kernel:__subpackages__",
        "@xla_serving//runtime/tensorflow:__subpackages__",
        "@xla_serving//runtime/xla_serving_data:__subpackages__",
    ],
    deps = [
        ":orbax_type_utils",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
        "@com_google_protobuf//:differencer",
        "@com_google_protobuf//:protobuf",
        "@com_google_protobuf//:protobuf_lite",
        "@tsl//tsl/platform:path",
        "@xla//xla:shape_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/ir:hlo_sharding",
        "@xla//xla/pjrt/proto:compile_options_proto_cc",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt:pjrt_dtype",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
        "@xla_serving//protos:manifest_cc_proto",
        "@xla_serving//protos:orchestration_cc_proto",
        "@xla_serving//protos:type_cc_proto",
        "@xla_serving//runtime/core:constants",
        "@xla_serving//runtime/core:flat_signature",
        "@xla_serving//runtime/core:stablehlo_utils",
        "@xla_serving//runtime/core:type",
        "@xla_serving//runtime/kernel:kernel_builder_context",
    ],
)

cc_library(
    name = "linear_graph_execute",
    srcs = ["linear_graph_execute.cc"],
    hdrs = ["linear_graph_execute.h"],
    visibility = ["@xla_serving//runtime/kernel/batching:__pkg__"],
    deps = [
        ":context",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@tsl//tsl/profiler/lib:connected_traceme",
        "@tsl//tsl/profiler/lib:context_types_hdrs",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//xla/tsl/concurrency:future",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla_serving//runtime/core:value",
        "@xla_serving//runtime/kernel",
    ],
)

# This is the interpreter that can run JAX-only models. This target is TF-free.
# To run models with TensorFlow preprocessing and postprocessing, link the
# "tensorflow:interpreter_tensorflow_support" target.
cc_library(
    name = "interpreter_core",
    srcs = [
        "bundle_interpreter.cc",
        "bundle_interpreter.h",
        "interpreter.cc",
        "interpreter.h",
    ],
    hdrs = [],
    visibility = ["//visibility:private"],
    deps = [
        ":context",
        ":interpreter_create_options",
        ":linear_graph_execute",
        ":manifest_utils",
        ":optional_function_handler",
        ":optional_function_handler_factory",
        ":run_options",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/types:span",
        "@com_google_protobuf//:protobuf_lite",
        "@tsl//tsl/platform:blocking_counter",
        "@tsl//tsl/platform:context",
        "@tsl//tsl/platform:path",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/ifrt:serdes",
        "@xla//xla/python/pjrt_ifrt:xla_ifrt",
        "@xla//xla/tsl/concurrency:future",
        "@xla//xla/tsl/lib/monitoring:counter",
        "@xla//xla/tsl/lib/monitoring:counter_gauge",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
        "@xla_serving//protos:bundle_orchestration_cc_proto",
        "@xla_serving//protos:manifest_cc_proto",
        "@xla_serving//protos:orchestration_cc_proto",
        "@xla_serving//protos:type_cc_proto",
        "@xla_serving//runtime/core:common_util",
        "@xla_serving//runtime/core:constants",
        "@xla_serving//runtime/core:flat_signature",
        "@xla_serving//runtime/core:ifrt_device_utils",
        "@xla_serving//runtime/core:ifrt_loaded_variable_registry",
        "@xla_serving//runtime/core:stablehlo",
        "@xla_serving//runtime/core:stablehlo_utils",
        "@xla_serving//runtime/core:value",
        "@xla_serving//runtime/core/batching:batch_scheduler",
        "@xla_serving//runtime/core/streaming",
        "@xla_serving//runtime/data_loader",
        "@xla_serving//runtime/kernel",
        "@xla_serving//runtime/kernel:kernel_builder_context",
        "@xla_serving//runtime/kernel:kernel_io_data_type",
        "@xla_serving//runtime/kernel/batching:batch_options_utils",
        "@xla_serving//runtime/kernel/batching:batch_resource_base",
        "@xla_serving//runtime/kernel/batching:batch_runner_op",
        "@xla_serving//runtime/libs/ifrt/persistent_compilation_cache:ifrt_persistent_compilation_cache",  # buildcleaner: keep
    ],
    alwayslink = 1,
)

cc_library(
    name = "checkpoint_weight_loading",
    visibility = [
        ":internal",
        ":interpreter_users",
    ],
    deps = [
        "@xla_serving//runtime/data_loader:default_data_loader_registration",
    ],
    alwayslink = 1,
)

cc_library(
    name = "jit_compilation",
    visibility = [
        ":internal",
        ":interpreter_users",
    ],
    deps = [
        ":init_lib",
    ],
    alwayslink = 1,
)

# This is the interpreter that can run JAX-only models. This target is TF-free.
# To run models with TensorFlow preprocessing and postprocessing, link the
# "tensorflow:interpreter_tensorflow_support" target.
cc_library(
    name = "interpreter",
    hdrs = [
        "bundle_interpreter.h",
        "interpreter.h",
    ],
    visibility = [
        ":internal",
        ":interpreter_users",
    ],
    deps = [
        ":checkpoint_weight_loading",
        ":context",
        ":interpreter_core",
        ":interpreter_create_options",
        ":jit_compilation",
        ":manifest_utils",
        ":optional_function_handler_factory",
        ":run_options",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/types:span",
        "@com_google_protobuf//:protobuf_lite",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/concurrency:future",
        "@xla//xla/tsl/platform:env",
        "@xla_serving//protos:bundle_orchestration_cc_proto",
        "@xla_serving//protos:manifest_cc_proto",
        "@xla_serving//protos:orchestration_cc_proto",
        "@xla_serving//protos:type_cc_proto",
        "@xla_serving//runtime/core:flat_signature",
        "@xla_serving//runtime/core:ifrt_loaded_variable_registry",
        "@xla_serving//runtime/core:stablehlo",
        "@xla_serving//runtime/core:value",
        "@xla_serving//runtime/data_loader",
        "@xla_serving//runtime/kernel",
        "@xla_serving//runtime/kernel:kernel_io_data_type",
    ],
)

cc_library(
    name = "run_options",
    hdrs = ["run_options.h"],
    visibility = [
        ":internal",
        ":interpreter_users",
    ],
    deps = [
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/time",
        "@xla_serving//runtime/core:host_buffer",
        "@xla_serving//runtime/core:stablehlo_utils",
    ],
)

cc_library(
    name = "context",
    hdrs = ["context.h"],
    # This is intended for internal implementation only and should not be exposed to users.
    visibility = [
        ":internal",
        "@xla_serving//runtime/core:__subpackages__",
        "@xla_serving//runtime/kernel:__subpackages__",
        "@xla_serving//runtime/tensorflow:__subpackages__",
        "@xla_serving//runtime/tensorflow/kernel:__pkg__",
    ],
    deps = [
        ":clock",
        ":run_options",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/time",
        "@xla_serving//runtime/core:host_buffer",
        "@xla_serving//runtime/core:stablehlo_utils",
    ],
)

cc_library(
    name = "clock",
    hdrs = ["clock.h"],
    visibility = [
        ":internal",
    ],
    deps = [
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/time",
    ],
)

cc_library(
    name = "init_lib",
    srcs = ["init_lib.cc"],
    visibility = [":internal"],
    deps = [
        "@abseil-cpp//absl/log:check",
        "@xla_serving//runtime/kernel",
        "@xla_serving//runtime/kernel:builtin_kernel",
    ],
    alwayslink = 1,
)

cc_library(
    name = "interpreter_create_options",
    hdrs = ["interpreter_create_options.h"],
    visibility = [
        ":internal",
        ":interpreter_users",
        "@xla_serving//runtime/xla_serving_data:__subpackages__",
    ],
    deps = [
        "@abseil-cpp//absl/base:nullability",
        "@tsl//tsl/platform:protobuf",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/platform:env",
        "@xla_serving//protos:manifest_cc_proto",
        "@xla_serving//runtime/core:allocator",
        "@xla_serving//runtime/core:default_allocator",
        "@xla_serving//runtime/core:stablehlo_optimizer",
        "@xla_serving//runtime/libs/ifrt/persistent_compilation_cache:ifrt_persistent_compilation_cache",
        "@xla_serving//runtime/libs/ifrt/serving_core_selector",
    ],
)

# The TF-compatible interpreter can run models with or without TensorFlow preprocessing.

# TODO: add a dependency check test to make sure that the interpreter does not depend
# on any TensorFlow library. Currently this is blocked. See cl/704297184 for more details.
