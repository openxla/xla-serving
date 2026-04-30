load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

config_setting(
    name = "enable_tf",
    values = {"define": "ENABLE_TF=1"},
)

cc_binary(
    name = "jsv_serving_main",
    srcs = ["main.cc"],
    linkopts = ["-Wl,--allow-multiple-definition"],
    deps = [
        ":jsv_callback_grpc_server",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/flags:parse",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@com_github_grpc_grpc//:grpc++",
        "@com_github_grpc_grpc//:grpc++_reflection",
        "@neptune//runtime/serving/gcp/auth:utils",
        "@xla//xla/tsl/platform:status",
    ],
)

cc_library(
    name = "grpc_status_utils",
    srcs = ["grpc_status_utils.cc"],
    hdrs = ["grpc_status_utils.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@com_github_grpc_grpc//:grpc++",
    ],
)

cc_library(
    name = "jsv_grpc_server",
    srcs = [
        "orbax_grpc_server.cc",
    ],
    hdrs = [
        "orbax_grpc_server.h",
    ],
    defines = select({
        ":enable_tf": ["ENABLE_TF=1"],
        "//conditions:default": [],
    }),
    deps = [
        ":flags",
        ":grpc_status_utils",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/time",
        "@com_github_grpc_grpc//:grpc++",
        "@neptune//runtime/core:host_buffer",
        "@neptune//runtime/core:value",
        "@neptune//runtime/core/proto:host_buffer_cc_proto",
        "@neptune//runtime/interpreter",
        "@neptune//runtime/interpreter:run_options",
        "@neptune//runtime/kernel:kernel_io_data_type",
        "@neptune//runtime/serving/gcp/proto:get_model_metadata_cc_proto",
        "@neptune//runtime/serving/gcp/proto:model_cc_proto",
        "@neptune//runtime/serving/gcp/proto:predict_cc_proto",
        "@neptune//runtime/serving/gcp/proto:prediction_service_cc_grpc_proto",
        "@neptune//runtime/serving/gcp/proto:prediction_service_cc_proto",
        "@neptune//runtime/tensorflow/utils:tf_to_host_buffer",
        "@org_tensorflow//tensorflow/core/framework:tensor",
        "@tsl//tsl/profiler/lib:traceme",
        "@tsl//tsl/profiler/lib:traceme_encode",
        "@xla//xla/pjrt:pjrt_api",
        "@xla//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "@xla//xla/pjrt/plugin/xla_cpu:cpu_static_registration",
        "@xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "@xla//xla/pjrt/plugin/xla_gpu:gpu_static_registration",
        "@xla//xla/pjrt/plugin/xla_gpu:xla_gpu_client_options",
        "@xla//xla/pjrt/plugin/xla_gpu:xla_gpu_pjrt_client",
        "@xla//xla/pjrt/plugin/xla_tpu:xla_tpu_pjrt_client",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ] + select({
        ":enable_tf": ["@neptune//runtime/tensorflow:tf_compatible_interpreter_create_options"],
        "//conditions:default": ["@neptune//runtime/interpreter:interpreter_create_options"],
    }),
)

cc_library(
    name = "jsv_callback_grpc_server",
    srcs = ["orbax_callback_grpc_server.cc"],
    hdrs = ["orbax_callback_grpc_server.h"],
    deps = [
        ":flags",
        ":grpc_status_utils",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/time",
        "@com_github_grpc_grpc//:grpc++",
        "@neptune//runtime/core:host_buffer",
        "@neptune//runtime/core:value",
        "@neptune//runtime/core/proto:host_buffer_cc_proto",
        "@neptune//runtime/interpreter",
        "@neptune//runtime/interpreter:run_options",
        "@neptune//runtime/kernel:kernel_io_data_type",
        "@neptune//runtime/serving/gcp/proto:get_model_metadata_cc_proto",
        "@neptune//runtime/serving/gcp/proto:model_cc_proto",
        "@neptune//runtime/serving/gcp/proto:predict_cc_proto",
        "@neptune//runtime/serving/gcp/proto:prediction_service_cc_grpc_proto",
        "@neptune//runtime/serving/gcp/proto:prediction_service_cc_proto",
        "@neptune//runtime/tensorflow:tf_compatible_interpreter_create_options",
        "@neptune//runtime/tensorflow/utils:tf_to_host_buffer",
        "@org_tensorflow//tensorflow/core/framework:tensor",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//xla/pjrt:pjrt_api",
        "@xla//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "@xla//xla/pjrt/plugin/xla_cpu:cpu_static_registration",
        "@xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "@xla//xla/pjrt/plugin/xla_gpu:gpu_static_registration",
        "@xla//xla/pjrt/plugin/xla_gpu:xla_gpu_client_options",
        "@xla//xla/pjrt/plugin/xla_gpu:xla_gpu_pjrt_client",
        "@xla//xla/pjrt/plugin/xla_tpu:xla_tpu_pjrt_client",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "flags",
    srcs = ["flags.cc"],
    hdrs = ["flags.h"],
    deps = [
        "@abseil-cpp//absl/flags:flag",
    ],
)
