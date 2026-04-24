load("@rules_cc//cc:cc_library.bzl", "cc_library")

package_group(
    name = "internal",
    packages = [
        "//learning/infra/mira/experimental/orbax_model/interpreter/...",
        "//runtime/interpreter/...",
    ],
)

package_group(
    name = "interpreter_users",
    packages = [
        "//learning/infra/mira/experimental/orbax_model/interpreter/...",
        "//learning/infra/mira/experimental/orbax_model/serving/...",
        "//runtime/...",
        "//smartass/brain/lemming/...",
        "//smartass/brain/server/...",
    ],
)

package(default_visibility = [
    ":internal",
])

cc_library(
    name = "optional_function_handler",
    hdrs = ["optional_function_handler.h"],
    visibility = [
        "//learning/infra/mira/experimental/orbax_model/tensorflow:__subpackages__",
        "@neptune//runtime/tensorflow:__subpackages__",
    ],
    deps = [
        ":interpreter_create_options",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@neptune//protos:manifest_cc_proto",
        "@neptune//protos:orchestration_cc_proto",
        "@neptune//runtime/kernel",
        "@neptune//runtime/kernel:kernel_io_data_type",
    ],
)

cc_library(
    name = "optional_function_handler_factory",
    srcs = ["optional_function_handler_factory.cc"],
    hdrs = ["optional_function_handler_factory.h"],
    visibility = [
        "//learning/infra/mira/experimental/orbax_model/tensorflow:__subpackages__",
        "@neptune//runtime/tensorflow:__subpackages__",
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
        "@neptune//protos:type_cc_proto",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/python/ifrt",
    ],
)

cc_library(
    name = "manifest_utils",
    srcs = ["manifest_utils.cc"],
    hdrs = ["manifest_utils.h"],
    visibility = [
        "//learning/infra/mira/experimental/orbax_model/interpreter:__subpackages__",
        "//learning/infra/mira/experimental/orbax_model/tensorflow:__subpackages__",
        "//learning/tfx/pipeline/infra_validation:__subpackages__",
        "//smartass/brain/server/graph_processing:__subpackages__",
        "@neptune//runtime/core:__subpackages__",
        "@neptune//runtime/data_loader:__subpackages__",
        "@neptune//runtime/kernel:__subpackages__",
        "@neptune//runtime/tensorflow:__subpackages__",
    ],
    deps = [
        ":orbax_type_utils",
        "@com_google_protobuf//:protobuf",
        "@com_google_protobuf//:protobuf_lite",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
        "@com_google_protobuf//:differencer",
        "@neptune//protos:manifest_cc_proto",
        "@neptune//protos:orchestration_cc_proto",
        "@neptune//protos:type_cc_proto",
        "@neptune//runtime/core:constants",
        "@neptune//runtime/core:flat_signature",
        "@neptune//runtime/core:stablehlo_utils",
        "@neptune//runtime/core:type",
        "@neptune//runtime/kernel:kernel_builder_context",
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
    ],
)

cc_library(
    name = "linear_graph_execute",
    srcs = ["linear_graph_execute.cc"],
    hdrs = ["linear_graph_execute.h"],
    visibility = ["@neptune//runtime/kernel/batching:__pkg__"],
    deps = [
        ":context",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@neptune//runtime/core:value",
        "@neptune//runtime/kernel",
        "@tsl//tsl/profiler/lib:connected_traceme",
        "@tsl//tsl/profiler/lib:context_types_hdrs",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//xla/tsl/concurrency:future",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
    ],
)

# This is the interpreter that can run JAX-only models. This target is TF-free.
# To run models with TensorFlow preprocessing and postprocessing, link the
# "//learning/infra/mira/experimental/orbax_model/tensorflow:interpreter_tensorflow_support" target.
cc_library(
    name = "interpreter",
    srcs = ["interpreter.cc"],
    hdrs = ["interpreter.h"],
    visibility = [
        ":internal",
        ":interpreter_users",
    ],
    deps = [
        ":context",
        ":init_lib",  # buildcleaner: keep
        ":interpreter_create_options",
        ":linear_graph_execute",
        ":manifest_utils",
        ":optional_function_handler",
        ":optional_function_handler_factory",
        ":run_options",
        "@com_google_protobuf//:protobuf_lite",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:nullability",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/memory",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/types:span",
        "@neptune//protos:manifest_cc_proto",
        "@neptune//protos:orchestration_cc_proto",
        "@neptune//protos:type_cc_proto",
        "@neptune//runtime/core:constants",
        "@neptune//runtime/core:flat_signature",
        "@neptune//runtime/core:ifrt_loaded_variable_registry",
        "@neptune//runtime/core:stablehlo",
        "@neptune//runtime/core:stablehlo_utils",
        "@neptune//runtime/core:value",
        "@neptune//runtime/core/batching:batch_scheduler",
        "@neptune//runtime/data_loader",
        "@neptune//runtime/data_loader:default_data_loader_registration",  # buildcleaner: keep
        "@neptune//runtime/kernel",
        "@neptune//runtime/kernel:kernel_builder_context",
        "@neptune//runtime/kernel:kernel_io_data_type",
        "@neptune//runtime/kernel/batching:batch_options_utils",
        "@neptune//runtime/kernel/batching:batch_runner_op",
        "@neptune//runtime/libs/ifrt/persistent_compilation_cache:ifrt_persistent_compilation_cache",  # buildcleaner: keep
        "@tsl//tsl/platform:context",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/concurrency:future",
        "@xla//xla/tsl/lib/monitoring:counter",
        "@xla//xla/tsl/lib/monitoring:counter_gauge",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "run_options",
    hdrs = ["run_options.h"],
    visibility = [
        ":internal",
        ":interpreter_users",
        "@neptune//runtime/kernel:__subpackages__",
    ],
    deps = [
        "@abseil-cpp//absl/time",
    ],
)

cc_library(
    name = "context",
    hdrs = ["context.h"],
    # This is intended for internal implementation only and should not be exposed to users.
    visibility = [
        ":internal",
        "//learning/infra/mira/experimental/orbax_model/tensorflow/kernel:__subpackages__",
        "@neptune//runtime/kernel:__subpackages__",
        "@neptune//runtime/tensorflow:__subpackages__",
        "@neptune//runtime/tensorflow/kernel:__pkg__",
    ],
    deps = [
        ":clock",
        ":run_options",
        "@abseil-cpp//absl/time",
    ],
)

cc_library(
    name = "clock",
    hdrs = ["clock.h"],
    visibility = [
        ":internal",
        "//learning/infra/mira/experimental/orbax_model/tensorflow/kernel:__subpackages__",
        "@neptune//runtime/kernel:__subpackages__",
        "@neptune//runtime/tensorflow:__subpackages__",
        "@neptune//runtime/tensorflow/kernel:__pkg__",
    ],
    deps = [
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/time",
    ],
)

cc_library(
    name = "init_lib",
    srcs = ["init_lib.cc"],
    visibility = [":interpreter_users"],
    deps = [
        "@abseil-cpp//absl/log:check",
        "@neptune//runtime/kernel",
        "@neptune//runtime/kernel:builtin_kernel",
    ],
    alwayslink = 1,
)

cc_library(
    name = "interpreter_create_options",
    hdrs = ["interpreter_create_options.h"],
    visibility = [
        ":internal",
        ":interpreter_users",
        # For TF-compatible interpreter to add more TF-specific options.
        "//learning/infra/mira/experimental/orbax_model/tensorflow:__subpackages__",
    ],
    deps = [
        "@abseil-cpp//absl/base:nullability",
        "@neptune//protos:manifest_cc_proto",
        "@neptune//runtime/core:allocator",
        "@neptune//runtime/core:default_allocator",
        "@neptune//runtime/core:stablehlo_optimizer",
        "@neptune//runtime/libs/ifrt/persistent_compilation_cache:ifrt_persistent_compilation_cache",
        "@tsl//tsl/platform:protobuf",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/platform:env",
    ],
)

# The TF-compatible interpreter can run models with or without TensorFlow preprocessing.

# TODO: add a dependency check test to make sure that the interpreter does not depend
# on any TensorFlow library. Currently this is blocked. See cl/704297184 for more details.
