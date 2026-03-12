load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//devtools/deps/check:deps_check.bzl", "check_dependencies")

cc_library(
    name = "batch_resource_base",
    srcs = ["batch_resource_base.cc"],
    hdrs = ["batch_resource_base.h"],
    visibility = [
        "@neptune//runtime/kernel:__subpackages__",
    ],
    deps = [
        ":warmup",
        "//third_party/tensorflow/core/common_runtime:cost_constants",
        "//third_party/tensorflow/core/common_runtime:cost_measurement",
        "//third_party/tensorflow/core/common_runtime:cost_util",
        "//third_party/tensorflow/core/common_runtime:request_cost",
        "//third_party/tensorflow/core/common_runtime:request_cost_accessor",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@neptune//runtime/core:allocator",
        "@neptune//runtime/core:host_buffer",
        "@neptune//runtime/core:incremental_barrier",
        "@neptune//runtime/core:value",
        "@neptune//runtime/core/batching:batch_scheduler",
        "@neptune//runtime/core/batching:batch_scheduler_utils",
        "@neptune//runtime/core/batching:batch_stats",
        "@neptune//runtime/core/batching:input_split_metadata",
        "@neptune//runtime/core/batching:resource_base",
        "@neptune//runtime/core/batching:shared_batch_scheduler",
        "@neptune//runtime/core/batching:threadsafe_status",
        "@neptune//runtime/kernel",
        "@tsl//tsl/platform:context",
        "@tsl//tsl/platform:thread_annotations",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/lib/monitoring:counter",
        "@xla//xla/tsl/lib/monitoring:sampler",
        "@xla//xla/tsl/platform:criticality",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:env_time",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "batch_runner_op",
    srcs = ["batch_runner_op.cc"],
    hdrs = ["batch_runner_op.h"],
    visibility = [
        "@neptune//runtime/interpreter:__pkg__",
        "@neptune//runtime/kernel:__subpackages__",
    ],
    deps = [
        ":batch_options_utils",
        ":batch_resource_base",
        ":warmup",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@neptune//runtime/core:allocator",
        "@neptune//runtime/core:default_allocator",
        "@neptune//runtime/core:host_buffer",
        "@neptune//runtime/core:value",
        "@neptune//runtime/core:value_utils",
        "@neptune//runtime/core/batching:batch_scheduler",
        "@neptune//runtime/core/batching:shared_batch_scheduler",
        "@neptune//runtime/interpreter:linear_graph_execute",
        "@neptune//runtime/kernel",
        "@neptune//runtime/kernel:kernel_builder_context",
        "@neptune//runtime/kernel:kernel_io_data_type",
        "@tsl//tsl/platform:random",
        "@tsl//tsl/platform:refcount",
        "@tsl//tsl/profiler/lib:connected_traceme",
        "@tsl//tsl/profiler/lib:context_types_hdrs",
        "@tsl//tsl/profiler/lib:traceme",
        "@tsl//tsl/profiler/lib:traceme_encode",
        "@xla//xla/tsl/concurrency:future",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
    ],
)

check_dependencies(
    name = "batch_runner_op_deps_test",
    of = [":batch_runner_op"],
    allowed_match_regexp = "|".join([
        # go/keep-sorted start
        "//third_party/tensorflow/compiler/xla(/|:)",  # OpenXLA
        "//third_party/tensorflow/core/common_runtime:.*cost.*",  # cost measurement libs.
        "//third_party/tensorflow/core/platform(/|:)",  # contains data types. Brought in by cost measurement libs.
        "//third_party/tensorflow/core:platform_base",  # brought in by "request_cost_accessor_registry".
        "//third_party/tensorflow/core:tflite_portable_logging",  # brought in by "request_cost_accessor_registry".
        "//third_party/tensorflow/third_party(/|:)",
        "//third_party/tensorflow/tsl(/|:)",  # go/moving-tsl-into-xla-lsc
        "//third_party/tensorflow:",  # contains platform-related libs. Brought in by several libs.
        # go/keep-sorted end
    ]),
    dont_match_regexp = "//third_party/tensorflow",
    error_context = "'BatchRunnerOp' should not have any dependency on TensorFlow.",
)

cc_library(
    name = "warmup",
    srcs = ["warmup.cc"],
    hdrs = ["warmup.h"],
    visibility = ["//learning/serving/servables/tfrt:__pkg__"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "batch_options_utils",
    srcs = ["batch_options_utils.cc"],
    hdrs = ["batch_options_utils.h"],
    visibility = [
        "@neptune//runtime/core/batching:__subpackages__",
        "@neptune//runtime/interpreter:__subpackages__",
        "@neptune//runtime/kernel/batching:__subpackages__",
    ],
    deps = [
        "//third_party/py/orbax/export/protos:oex_orchestration_cc_proto",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@neptune//runtime/core/batching:batch_scheduler",
    ],
)

cc_library(
    name = "test_batch_resource_base",
    testonly = True,
    hdrs = ["test_batch_resource_base.h"],
    visibility = [
        "@neptune//runtime/kernel:__subpackages__",
    ],
    deps = [
        ":batch_resource_base",
        ":batch_runner_op",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@neptune//runtime/core:host_buffer",
        "@neptune//runtime/core/batching:shared_batch_scheduler",
        "@neptune//runtime/kernel",
        "@xla//xla/tsl/platform:env",
    ],
)
