load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = [
    "//learning/infra/mira/experimental/orbax_model/tensorflow:__subpackages__",
    "//learning/serving/servables/tfrt:__subpackages__",
    "//smartass/brain/server:__subpackages__",
    "@neptune//runtime/python:__subpackages__",
    "@neptune//runtime/serving:__subpackages__",
    "@neptune//runtime/tensorflow:__subpackages__",
])

cc_library(
    name = "common_utils",
    srcs = ["common_utils.cc"],
    hdrs = ["common_utils.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
        "@neptune//protos:manifest_cc_proto",
        "@neptune//runtime/core:constants",
        "@neptune//runtime/interpreter:manifest_utils",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "ifrt_tensor_utils",
    srcs = ["ifrt_tensor_utils.cc"],
    hdrs = ["ifrt_tensor_utils.h"],
    deps = [
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@xla//xla:shape_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt:pjrt_dtype",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "tf_to_host_buffer",
    srcs = ["tf_to_host_buffer.cc"],
    hdrs = ["tf_to_host_buffer.h"],
    deps = [
        ":ifrt_tensor_utils",
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@neptune//runtime/core:host_buffer",
        "@neptune//runtime/core:value",
        "@neptune//runtime/kernel:kernel_io_data_type",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@tsl//tsl/platform:refcount",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "tf_to_jsv_proto",
    srcs = ["tf_to_jsv_proto.cc"],
    hdrs = ["tf_to_jsv_proto.h"],
    deps = [
        ":ifrt_tensor_utils",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@neptune//runtime/core/proto:host_buffer_cc_proto",
        "@neptune//runtime/serving/gcp/proto:model_cc_proto",
        "@neptune//runtime/serving/gcp/proto:predict_cc_proto",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/framework:tensor",
        "@tsl//tsl/platform:tstring",
        "@xla//xla/python/ifrt:dtype_proto_cc",
        "@xla//xla/tsl/platform:statusor",
    ],
)
