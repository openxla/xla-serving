load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = [
    "@neptune//runtime/interpreter:__pkg__",
    "@neptune//runtime/kernel/batching:__subpackages__",
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

cc_library(
    name = "shared_batch_scheduler",
    hdrs = [
        "fifo_queue.h",
        "queue_base.h",
        "shared_batch_scheduler.h",
    ],
    deps = [
        ":batch_scheduler",
        ":batch_scheduler_utils",
        ":batch_stats",
        ":periodic_function",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@neptune//runtime/core:allocator",
        "@neptune//runtime/core:default_allocator",
        "@neptune//runtime/kernel/batching:batch_options_utils",
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
        "@neptune//runtime/kernel/batching:batch_options_utils",
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

cc_library(
    name = "input_split_metadata",
    srcs = ["input_split_metadata.cc"],
    hdrs = ["input_split_metadata.h"],
    deps = [
        "@abseil-cpp//absl/container:fixed_array",
        "@abseil-cpp//absl/strings",
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
