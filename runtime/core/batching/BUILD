load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = [
    # TODO: Revert to :__pkg__ once the migration to QueueBase is complete.
    "@xla_serving//runtime/interpreter:__subpackages__",
    "@xla_serving//runtime/kernel/batching:__subpackages__",
])

cc_library(
    name = "batch_scheduler",
    srcs = ["batch_scheduler.cc"],
    hdrs = ["batch_scheduler.h"],
    deps = [
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/synchronization",
        "@tsl//tsl/platform:thread_annotations",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//xla/tsl/platform:criticality",
    ],
)

cc_test(
    name = "batch_scheduler_test",
    srcs = ["batch_scheduler_test.cc"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS"],
    deps = [
        ":batch_scheduler",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/synchronization",
        "@googletest//:gtest_main",
        "@xla//xla/tsl/platform:criticality",
        "@xla//xla/tsl/platform:env",
    ],
)

cc_library(
    name = "queue_options",
    hdrs = ["queue_options.h"],
    deps = [
        ":batch_scheduler",
        ":batch_stats",
        "@abseil-cpp//absl/status",
        "@xla_serving//runtime/core:allocator",
        "@xla_serving//runtime/core:default_allocator",
        "@xla_serving//runtime/kernel/batching:batch_options_utils",
    ],
)

cc_library(
    name = "queue_base",
    hdrs = ["queue_base.h"],
    deps = [
        ":batch_scheduler",
        ":queue_options",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/synchronization",
        "@tsl//tsl/profiler/lib:connected_traceme",
        "@tsl//tsl/profiler/lib:context_types_hdrs",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
    ],
)

cc_library(
    name = "fifo_queue",
    hdrs = ["fifo_queue.h"],
    deps = [
        ":batch_scheduler",
        ":batch_scheduler_utils",
        ":queue_base",
        ":queue_options",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@tsl//tsl/profiler/lib:connected_traceme",
        "@tsl//tsl/profiler/lib:context_types_hdrs",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
    ],
)

cc_library(
    name = "priority_aware_queue",
    hdrs = ["priority_aware_queue.h"],
    deps = [
        ":batch_scheduler",
        ":batch_scheduler_utils",
        ":queue_base",
        ":queue_options",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@xla//xla/tsl/platform:criticality",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
    ],
)

cc_library(
    name = "shared_batch_scheduler",
    srcs = ["shared_batch_scheduler.cc"],
    hdrs = ["shared_batch_scheduler.h"],
    deps = [
        ":batch_scheduler",
        ":batch_scheduler_utils",
        ":fifo_queue",
        ":periodic_function",
        ":priority_aware_queue",
        ":queue_base",
        ":queue_options",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:thread_annotations",
        "@tsl//tsl/profiler/lib:connected_traceme",
        "@tsl//tsl/profiler/lib:context_types_hdrs",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//xla/tsl/platform:criticality",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
    ],
)

cc_test(
    name = "shared_batch_scheduler_test",
    srcs = ["shared_batch_scheduler_test.cc"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS"],
    deps = [
        ":batch_scheduler",
        ":fake_clock_env",
        ":input_split_metadata",
        ":shared_batch_scheduler",
        "@abseil-cpp//absl/base",
        "@abseil-cpp//absl/container:fixed_array",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
        "@tsl//tsl/platform:platform_port",
        "@xla//xla/tsl/lib/core:status_test_util",
        "@xla//xla/tsl/platform:criticality",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:status",
        "@xla//xla/tsl/platform:test_benchmark",
        "@xla_serving//runtime/kernel/batching:batch_options_utils",
    ],
)

cc_test(
    name = "fifo_queue_test",
    srcs = ["fifo_queue_test.cc"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS"],
    deps = [
        ":batch_scheduler",
        ":fake_clock_env",
        ":fifo_queue",
        ":input_split_metadata",
        ":queue_options",
        "@abseil-cpp//absl/container:fixed_array",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
        "@xla//xla/tsl/lib/core:status_test_util",
        "@xla//xla/tsl/platform:criticality",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:status",
    ],
)

cc_test(
    name = "priority_aware_queue_test",
    srcs = ["priority_aware_queue_test.cc"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS"],
    deps = [
        ":batch_scheduler",
        ":fake_clock_env",
        ":input_split_metadata",
        ":priority_aware_queue",
        ":queue_options",
        "@abseil-cpp//absl/container:fixed_array",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@googletest//:gtest_main",
        "@xla//xla/tsl/platform:criticality",
        "@xla//xla/tsl/platform:env",
        "@xla_serving//runtime/kernel/batching:batch_options_utils",
    ],
)

cc_library(
    name = "batch_scheduler_utils",
    srcs = ["batch_scheduler_utils.cc"],
    hdrs = ["batch_scheduler_utils.h"],
    deps = [
        ":batch_scheduler",
        ":batch_stats",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/time",
        "@xla_serving//runtime/kernel/batching:batch_options_utils",
    ],
)

cc_test(
    name = "batch_scheduler_utils_test",
    srcs = ["batch_scheduler_utils_test.cc"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS"],
    deps = [
        ":batch_scheduler",
        ":batch_scheduler_utils",
        ":batch_stats",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
        "@xla_serving//runtime/kernel/batching:batch_options_utils",
    ],
)

cc_library(
    name = "batch_stats",
    hdrs = ["batch_stats.h"],
    deps = [
        "@abseil-cpp//absl/container:node_hash_map",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@tsl//tsl/platform:thread_annotations",
    ],
)

cc_test(
    name = "batch_stats_test",
    srcs = ["batch_stats_test.cc"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS"],
    deps = [
        ":batch_stats",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "fake_clock_env",
    srcs = ["fake_clock_env.cc"],
    hdrs = ["fake_clock_env.h"],
    deps = [
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/synchronization",
        "@tsl//tsl/platform:thread_annotations",
        "@xla//xla/tsl/platform:env",
    ],
)

cc_library(
    name = "periodic_function",
    srcs = ["periodic_function.cc"],
    hdrs = ["periodic_function.h"],
    deps = [
        "@abseil-cpp//absl/functional:any_invocable",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@xla//xla/tsl/platform:env",
    ],
)

cc_test(
    name = "periodic_function_test",
    srcs = ["periodic_function_test.cc"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS"],
    deps = [
        ":fake_clock_env",
        ":periodic_function",
        "@abseil-cpp//absl/synchronization",
        "@googletest//:gtest_main",
        "@xla//xla/tsl/platform:env",
    ],
)

cc_library(
    name = "input_split_metadata",
    srcs = ["input_split_metadata.cc"],
    hdrs = ["input_split_metadata.h"],
    deps = [
        "@abseil-cpp//absl/container:fixed_array",
        "@abseil-cpp//absl/strings",
    ],
)

cc_test(
    name = "input_split_metadata_test",
    srcs = ["input_split_metadata_test.cc"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS"],
    deps = [
        ":input_split_metadata",
        "@abseil-cpp//absl/container:fixed_array",
        "@abseil-cpp//absl/strings",
        "@googletest//:gtest_main",
    ],
)

cc_library(
    name = "resource_base",
    hdrs = ["resource_base.h"],
    deps = [
        "@abseil-cpp//absl/strings:str_format",
        "@tsl//tsl/platform:refcount",
    ],
)

cc_library(
    name = "threadsafe_status",
    srcs = ["threadsafe_status.cc"],
    hdrs = ["threadsafe_status.h"],
    deps = [
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/synchronization",
    ],
)

cc_test(
    name = "threadsafe_status_test",
    srcs = ["threadsafe_status_test.cc"],
    defines = ["ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS"],
    deps = [
        ":threadsafe_status",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:status_matchers",
        "@googletest//:gtest_main",
    ],
)
