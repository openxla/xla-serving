load("@rules_cc//cc:cc_library.bzl", "cc_library")

# Limit visibility to selected packages.
package_group(
    name = "tf_function_internal_users",
    packages = [
        "//runtime/python/...",
        "//runtime/serving/...",
    ],
)

package_group(
    name = "tf_function_friends",
    packages = [
        "//runtime/tensorflow/kernel/...",
    ],
)

# Include this library in the `deps` of the `cc_binary` runtime/interpreter/BUILDto enable TF-based preprocessing and postprocessing.
cc_library(
    name = "tf_compatible_optional_function_handler",
    srcs = ["tf_compatible_optional_function_handler.cc"],
    hdrs = ["tf_compatible_optional_function_handler.h"],
    deps = [
        ":tf_compatible_interpreter_create_options",
        ":tf_saved_model_runner",
        "@com_google_protobuf//:protobuf_lite",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/base",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:string_view",
        "@neptune//protos:manifest_cc_proto",
        "@neptune//protos:orchestration_cc_proto",
        "@neptune//protos:tf_concrete_function_handle_cc_proto",
        "@neptune//protos:type_cc_proto",
        "@neptune//runtime/core:constants",
        "@neptune//runtime/core:flat_signature",
        "@neptune//runtime/interpreter:interpreter_create_options",
        "@neptune//runtime/interpreter:manifest_utils",
        "@neptune//runtime/interpreter:optional_function_handler",
        "@neptune//runtime/interpreter:optional_function_handler_factory",
        "@neptune//runtime/kernel",
        "@neptune//runtime/kernel:kernel_builder_context",
        "@neptune//runtime/kernel:kernel_io_data_type",
        "@neptune//runtime/tensorflow/kernel:kernel_builder_constants",
        "@neptune//runtime/tensorflow/utils:common_utils",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@tsl//tsl/platform:path",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
    alwayslink = True,
)

cc_library(
    name = "tf_saved_model_runner",
    srcs = ["tf_saved_model_runner.cc"],
    hdrs = ["tf_saved_model_runner.h"],
    visibility = [":tf_function_friends"],
    deps = [
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/types:span",
        "@org_tensorflow//tensorflow/cc/saved_model:tag_constants",
        "@org_tensorflow//tensorflow/core:core_cpu_base",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/framework:graph_proto_cc",
        "@org_tensorflow//tensorflow/core/tfrt/runtime",
        "@org_tensorflow//tensorflow/core/tfrt/runtime:work_queue_interface",
        "@org_tensorflow//tensorflow/core/tfrt/saved_model:saved_model_cpu",
        "@tf_runtime//:hostcontext",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "tf_compatible_interpreter_create_options",
    hdrs = ["tf_compatible_interpreter_create_options.h"],
    visibility = [
        ":tf_function_internal_users",
        "@neptune//runtime/interpreter:interpreter_users",
    ],
    deps = [
        "@neptune//runtime/interpreter:interpreter_create_options",
        "@org_tensorflow//tensorflow/core/tfrt/saved_model:saved_model_cpu",
    ],
)

# Include this library in the `deps` of the `cc_binary` to enable TF-based preprocessing and postprocessing.
cc_library(
    name = "interpreter_tensorflow_support",
    visibility = [
        ":tf_function_internal_users",
        "@neptune//runtime/interpreter:interpreter_users",
    ],
    deps = [
        ":tf_compatible_optional_function_handler",
        "@neptune//runtime/tensorflow/kernel:init_lib_tensorflow",
    ],
)

cc_library(
    name = "tfrt_thread_pool",
    hdrs = ["tfrt_thread_pool.h"],
    deps = [
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@eigen_archive//:eigen3",
        "@neptune//runtime/core:thread_pool_interface",
        "@org_tensorflow//tensorflow/core/tfrt/runtime:work_queue_interface",
        "@tf_runtime//:hostcontext",
    ],
)
