load("@aspect_bazel_lib//lib:tar.bzl", "tar")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

# The following configuration is only for the public OSS version.
# import oss_only_module
load("@rules_oci//oci:defs.bzl", "oci_image", "oci_push")

config_setting(
    name = "enable_tf",
    values = {"define": "ENABLE_TF=1"},
)

XLA_SERVING_MAIN_COMMON_DEPS = [
    "@abseil-cpp//absl/flags:flag",
    "@abseil-cpp//absl/flags:parse",
    "@abseil-cpp//absl/log",
    "@abseil-cpp//absl/log:check",
    "@abseil-cpp//absl/status:statusor",
    "@abseil-cpp//absl/strings",
    "@grpc//:grpc++",
    "@grpc//:grpc++_reflection",
    "@xla_serving//protos/grpc:model_server_config_cc_proto",
    "@xla_serving//runtime/serving/gcp/auth:utils",
    "@com_google_protobuf//:protobuf",
    "@xla//xla/tsl/platform:env",
    "@xla//xla/tsl/platform:status",
    "@tsl//tsl/platform:platform_port",
    "@xla//xla/tsl/profiler/rpc:profiler_server_impl",
]

cc_binary(
    name = "xla_serving_main_cpu",
    srcs = ["main.cc"],
    linkopts = ["-Wl,--allow-multiple-definition"],
    deps = [
        ":jsv_callback_grpc_server_cpu",
    ] + XLA_SERVING_MAIN_COMMON_DEPS,
)

cc_binary(
    name = "xla_serving_main_gpu",
    srcs = ["main.cc"],
    linkopts = ["-Wl,--allow-multiple-definition"],
    tags = ["manual"],
    deps = [
        ":jsv_callback_grpc_server_gpu",
    ] + XLA_SERVING_MAIN_COMMON_DEPS,
)

cc_binary(
    name = "xla_serving_main_tpu",
    srcs = ["main.cc"],
    linkopts = ["-Wl,--allow-multiple-definition"],
    tags = ["manual"],
    deps = [
        ":jsv_callback_grpc_server_tpu",
    ] + XLA_SERVING_MAIN_COMMON_DEPS,
)

# The following configuration is only for the public OSS version.
# import oss_only_module
# Package the binary in tar layer.
tar(
    name = "xla_serving",
    srcs = [":xla_serving_main_cpu"],
    tags = ["manual"],
)

# 3. Define the Image
oci_image(
    name = "xla_serving_image",
    base = "@distroless_cc",
    entrypoint = ["/runtime/serving/gcp/server/xla_serving_main_cpu"],
    tags = ["manual"],
    tars = [":xla_serving"],
)

# 4. Define the Push destination, e.g. command:
# bazel run //runtime/serving/gcp/server:xla_serving_pusher -- --repository=us-docker.pkg.dev/my-project/my-repo/image
oci_push(
    name = "xla_serving_pusher",
    image = ":xla_serving_image",
    repository = "gcr.io/neptune-serving-engine/cpu-jax-only",
    tags = ["manual"],
)

# Package the GPU binary in tar layer.
tar(
    name = "xla_serving_gpu",
    srcs = [":xla_serving_main_gpu"],
    tags = ["manual"],
)

# Define the GPU Image
oci_image(
    name = "xla_serving_image_gpu",
    base = "@cuda_base",
    entrypoint = ["/runtime/serving/gcp/server/xla_serving_main_gpu"],
    tags = ["manual"],
    tars = [":xla_serving_gpu"],
)

# Define the GPU Push destination
oci_push(
    name = "xla_serving_pusher_gpu",
    image = ":xla_serving_image_gpu",
    repository = "gcr.io/neptune-serving-engine/gpu-jax",
    tags = ["manual"],
)

# Package the TPU binary in tar layer.
tar(
    name = "xla_serving_tpu",
    srcs = [":xla_serving_main_tpu"],
    tags = ["manual"],
)

# Define the TPU Image
oci_image(
    name = "xla_serving_image_tpu",
    base = "@distroless_cc",
    entrypoint = ["/runtime/serving/gcp/server/xla_serving_main_tpu"],
    tags = ["manual"],
    tars = [":xla_serving_tpu"],
)

# Define the TPU Push destination
oci_push(
    name = "xla_serving_pusher_tpu",
    image = ":xla_serving_image_tpu",
    repository = "gcr.io/neptune-serving-engine/tpu-jax",
    tags = ["manual"],
)

cc_library(
    name = "grpc_status_utils",
    srcs = ["grpc_status_utils.cc"],
    hdrs = ["grpc_status_utils.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@grpc//:grpc++",
    ],
)

JSV_CALLBACK_GRPC_SERVER_COMMON_DEPS = [
    ":flags",
    ":grpc_status_utils",
    "@abseil-cpp//absl/cleanup",
    "@abseil-cpp//absl/container:flat_hash_map",
    "@abseil-cpp//absl/flags:flag",
    "@abseil-cpp//absl/log",
    "@abseil-cpp//absl/log:check",
    "@abseil-cpp//absl/status",
    "@abseil-cpp//absl/status:statusor",
    "@abseil-cpp//absl/strings",
    "@abseil-cpp//absl/time",
    "@grpc//:grpc++",
    "@xla_serving//runtime/core:host_buffer",
    "@xla_serving//runtime/core:value",
    "@xla_serving//protos/grpc:host_buffer_cc_proto",
    "@xla_serving//runtime/interpreter",
    "@xla_serving//runtime/interpreter:run_options",
    "@xla_serving//runtime/kernel:kernel_io_data_type",
    "@xla_serving//protos/grpc:get_model_metadata_cc_proto",
    "@xla_serving//protos/grpc:model_cc_proto",
    "@xla_serving//protos/grpc:predict_cc_proto",
    "@xla_serving//protos/grpc:prediction_log_cc_proto",
    "@xla_serving//protos/grpc:prediction_service_cc_grpc_proto",
    "@xla_serving//protos/grpc:prediction_service_cc_proto",
    "@xla_serving//runtime/tensorflow/utils:tf_to_host_buffer",
    "@xla//xla/pjrt:pjrt_api",
    "@xla//xla/python/ifrt",
    "@xla//xla/python/pjrt_ifrt",
    "@xla//xla/tsl/concurrency:future",
    "@xla//xla/tsl/platform:env",
    "@xla//xla/tsl/platform:errors",
    "@xla//xla/tsl/platform:statusor",
    "@org_tensorflow//tensorflow/core/framework:tensor",
    "@org_tensorflow//tensorflow/core:lib",
    "@tsl//tsl/profiler/lib:traceme",
    "@xla_serving//protos/grpc:model_server_config_cc_proto",
] + select({
    ":enable_tf": [
        "@xla_serving//runtime/tensorflow:tf_compatible_interpreter_create_options",
        "@xla_serving//runtime/tensorflow:interpreter_tensorflow_support",
    ],
    "//conditions:default": ["@xla_serving//runtime/interpreter:interpreter_create_options"],
})

cc_library(
    name = "ifrt_client_creator_hdr",
    hdrs = ["ifrt_client_creator.h"],
    deps = [
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@xla//xla/python/ifrt",
    ],
)

cc_library(
    name = "ifrt_client_creator_cpu",
    srcs = ["ifrt_client_creator_cpu.cc"],
    deps = [
        ":ifrt_client_creator_hdr",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@xla//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "@xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt",
        "@xla//xla/tsl/platform:statusor",
    ],
    alwayslink = True,
)

cc_library(
    name = "ifrt_client_creator_gpu",
    srcs = ["ifrt_client_creator_gpu.cc"],
    tags = ["manual"],
    deps = [
        ":ifrt_client_creator_hdr",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@xla//xla/pjrt/plugin/xla_gpu:xla_gpu_client_options",
        "@xla//xla/pjrt/plugin/xla_gpu:xla_gpu_pjrt_client",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt",
        "@xla//xla/tsl/platform:statusor",
    ],
    alwayslink = True,
)

cc_library(
    name = "ifrt_client_creator_tpu",
    srcs = ["ifrt_client_creator_tpu.cc"],
    tags = ["manual"],
    deps = [
        ":ifrt_client_creator_hdr",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@xla//xla/pjrt/plugin/xla_tpu:xla_tpu_pjrt_client",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt",
        "@xla//xla/tsl/platform:statusor",
    ],
    alwayslink = True,
)

cc_library(
    name = "jsv_callback_grpc_server_cpu",
    srcs = ["orbax_callback_grpc_server.cc"],
    hdrs = ["orbax_callback_grpc_server.h"],
    defines = ["PLATFORM_CPU"] + select({
        ":enable_tf": ["ENABLE_TF=1"],
        "//conditions:default": [],
    }),
    deps = JSV_CALLBACK_GRPC_SERVER_COMMON_DEPS + [
        ":ifrt_client_creator_cpu",
        ":ifrt_client_creator_hdr",
        "@abseil-cpp//absl/types:span",
        "@com_google_protobuf//:delimited_message_util",
        "@xla//xla/pjrt/plugin/xla_cpu:cpu_static_registration",
    ],
)

cc_library(
    name = "jsv_callback_grpc_server_gpu",
    srcs = ["orbax_callback_grpc_server.cc"],
    hdrs = ["orbax_callback_grpc_server.h"],
    defines = ["PLATFORM_CUDA"] + select({
        ":enable_tf": ["ENABLE_TF=1"],
        "//conditions:default": [],
    }),
    tags = ["manual"],
    deps = JSV_CALLBACK_GRPC_SERVER_COMMON_DEPS + [
        ":ifrt_client_creator_gpu",
        ":ifrt_client_creator_hdr",
        "@abseil-cpp//absl/types:span",
        "@com_google_protobuf//:delimited_message_util",
        "@xla//xla/pjrt/plugin/xla_gpu:gpu_static_registration",
        "@xla//xla/pjrt/plugin/xla_gpu:xla_gpu_client_options",
        "@xla//xla/pjrt/plugin/xla_gpu:xla_gpu_pjrt_client",
        "@xla//xla/stream_executor/cuda:all_runtime",
    ],
)

cc_library(
    name = "jsv_callback_grpc_server_tpu",
    srcs = ["orbax_callback_grpc_server.cc"],
    hdrs = ["orbax_callback_grpc_server.h"],
    defines = ["PLATFORM_CLOUD_TPU"] + select({
        ":enable_tf": ["ENABLE_TF=1"],
        "//conditions:default": [],
    }),
    tags = ["manual"],
    deps = JSV_CALLBACK_GRPC_SERVER_COMMON_DEPS + [
        ":ifrt_client_creator_hdr",
        ":ifrt_client_creator_tpu",
        "@abseil-cpp//absl/types:span",
        "@com_google_protobuf//:delimited_message_util",
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

XLA_SERVING_MAIN_TEST_COMMON_DEPS = [
    ":flags",
    "//devtools/build/runtime:get_runfiles_dir",
    "//net/util:ports",
    "//testing/base/public:gunit_main",
    "@abseil-cpp//absl/flags:flag",
    "@abseil-cpp//absl/log",
    "@abseil-cpp//absl/strings",
    "@abseil-cpp//absl/time",
    "@grpc//:grpc++",
    "@xla_serving//protos/grpc:host_buffer_cc_proto",
    "@xla_serving//protos/grpc:model_cc_proto",
    "@xla_serving//protos/grpc:model_server_config_cc_proto",
    "@xla_serving//protos/grpc:predict_cc_proto",
    "@xla_serving//protos/grpc:prediction_service_cc_grpc_proto",
    "@xla_serving//runtime/core:host_buffer",
    "@xla_serving//runtime/core:host_buffer_matcher",
    "@com_google_protobuf//:protobuf",
    "@xla//xla/python/ifrt",
    "@xla//xla/tsl/platform:env",
    "@xla//xla/tsl/platform:status",
    "//util/process",
]
