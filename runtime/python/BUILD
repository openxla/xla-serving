load("//devtools/python/blaze:pytype.bzl", "pytype_strict_contrib_test", "pytype_strict_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//third_party/pybind11/google3_utils:build_defs.bzl", "pybind_extension")

package_group(
    name = "friends",
    packages = [
        "//learning/gemini/deployment/easyserve/exportable/...",  # @johnqiangzhang
        "//smartass/brain/lemming/...",
        "//third_party/py/orbax/export/...",
        "//video/youtube/discovery/ml_infra/jax/exporter/...",
        "//video/youtube/discovery/tensorflow/clients/ml_infra/jax/export/...",
    ],
)

package_group(
    name = "internal",
    packages = [
        "@neptune//runtime/python/...",
    ],
)

package(default_visibility = [
    ":internal",
])

pybind_extension(
    name = "orbax_model_runner",
    srcs = ["orbax_model_runner.cc"],
    py_deps = ["//third_party/py/jax"],
    visibility = [
        ":friends",
        ":internal",
    ],
    deps = [
        ":ifrt_client_factory",
        ":numpy_utils",
        "//learning/infra/mira/experimental/orbax_model/tensorflow:interpreter_tensorflow_support",  # buildcleaner: keep
        "//learning/infra/mira/experimental/orbax_model/tensorflow:tf_compatible_interpreter_create_options",
        "//third_party/nanobind",
        "//third_party/tensorflow/core:framework",
        "//third_party/tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@neptune//runtime/core:host_buffer",
        "@neptune//runtime/core:value",
        "@neptune//runtime/interpreter",
        "@neptune//runtime/kernel:kernel_io_data_type",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/python:nb_numpy",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/python/lib/core:numpy",
    ],
)

pytype_strict_library(
    name = "orbax_model_runner_test_base",
    testonly = True,
    srcs = ["orbax_model_runner_test_base.py"],
    data = [
        "//learning/infra/mira/experimental/orbax_model/interpreter/testdata:linear_model",
        "//learning/infra/mira/experimental/orbax_model/interpreter/testdata:make_number_from_digits_dict",
        "//learning/infra/mira/experimental/orbax_model/interpreter/testdata:multiple_apply_fn_sharing_ckpt",
        "//learning/infra/mira/experimental/orbax_model/interpreter/testdata:simple_model_with_tf_processing",
        "//learning/infra/mira/experimental/orbax_model/interpreter/testdata:string_to_number_round_trip_batch_model_func",
        "//learning/infra/mira/experimental/orbax_model/interpreter/testdata:string_to_number_round_trip_batch_model_func_and_post_processor",
        "//learning/infra/mira/experimental/orbax_model/interpreter/testdata:string_to_number_round_trip_batch_pre_processor_and_model_func",
        "//learning/infra/mira/experimental/orbax_model/interpreter/testdata:string_to_number_round_trip_batch_whole_pipeline",
        "//learning/infra/mira/experimental/orbax_model/interpreter/testdata:string_to_number_round_trip_no_batching",
    ],
    deps = [
        ":orbax_model_runner",
        "//pyglib:resources",
        "//testing/pybase",
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
    ],
)

pytype_strict_contrib_test(
    name = "orbax_model_runner_test",
    size = "small",
    testonly = True,
    srcs = ["orbax_model_runner_test.py"],
    args = [
        "--orbax_model_runner_platform=cpu",
    ],
    env = {
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
    },
    deps = [
        ":orbax_model_runner",
        ":orbax_model_runner_test_base",
        "//pyglib:resources",
        "//testing/pybase",
    ],
)

pytype_strict_contrib_test(
    name = "orbax_model_runner_test_cuda",
    size = "small",
    testonly = True,
    srcs = ["orbax_model_runner_test_cuda.py"],
    args = [
        "--orbax_model_runner_platform=cuda",
    ],
    data = [
        "//learning/infra/mira/experimental/orbax_model/interpreter/testdata:linear_model_2x1_sharded",
    ],
    tags = ["requires-gpu-nvidia:2"],
    deps = [
        ":orbax_model_runner",
        "//pyglib:resources",
        "//testing/pybase",
        "//third_party/py/numpy",
        "@xla//xla/pjrt/plugin/xla_gpu:gpu_static_registration",
    ],
)

cc_library(
    name = "numpy_utils",
    srcs = ["numpy_utils.cc"],
    hdrs = ["numpy_utils.h"],
    copts = ["-fexceptions"],  # go/cpp-3p-exceptions
    features = ["-use_header_modules"],  # Required for -fexceptions.
    deps = [
        "//learning/infra/mira/experimental/orbax_model/tensorflow/utils:ifrt_tensor_utils",
        "//third_party/nanobind",
        "//third_party/py/numpy:headers",
        "//third_party/python_runtime:headers",
        "//third_party/tensorflow/core:framework",
        "//third_party/tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@neptune//runtime/core:allocator",
        "@neptune//runtime/core:default_allocator",
        "@neptune//runtime/core:host_buffer",
        "@tsl//tsl/platform:tstring",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/python:nb_numpy",
        "@xla//xla/python:types",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/python/lib/core:numpy",
    ],
)

pybind_extension(
    name = "numpy_utils_test_utils",
    testonly = True,
    srcs = ["numpy_utils_test_utils.cc"],
    import_test = False,
    deps = [
        ":numpy_utils",
        "//third_party/nanobind",
        "//third_party/tensorflow/core:framework",
        "//third_party/tensorflow/core:portable_gif_internal",
        "//third_party/tensorflow/core:protos_all_cc",
        "@neptune//runtime/core:host_buffer",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/python:nb_numpy",
        "@xla//xla/tsl/python/lib/core:ml_dtypes_lib",  # buildcleaner: keep
        "@xla//xla/tsl/python/lib/core:numpy",
    ],
)

pytype_strict_contrib_test(
    name = "numpy_utils_test",
    size = "small",
    srcs = ["numpy_utils_test.py"],
    deps = [
        ":numpy_utils_test_utils",
        "//testing/pybase",
        "//testing/pybase:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

cc_library(
    name = "ifrt_client_factory",
    srcs = ["ifrt_client_factory.cc"],
    hdrs = ["ifrt_client_factory.h"],
    visibility = [":internal"],
    deps = [
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@xla//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "@xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "@xla//xla/pjrt/plugin/xla_gpu:xla_gpu_client_options",
        "@xla//xla/pjrt/plugin/xla_gpu:xla_gpu_pjrt_client",
        "@xla//xla/pjrt/plugin/xla_tpu:xla_tpu_pjrt_client",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/pjrt_ifrt",
        "@xla//xla/tsl/platform:statusor",
    ],
)
