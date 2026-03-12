load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//third_party/gpus/cuda:build_defs.bzl", "if_cuda")

package_group(
    name = "internal",
    packages = [
        "//learning/infra/mira/experimental/orbax_model/interpreter/...",
        "@neptune//runtime/interpreter/...",
    ],
)

package_group(
    name = "interpreter_users",
    packages = [
        "//learning/infra/mira/experimental/orbax_model/interpreter/...",
        "//learning/infra/mira/experimental/orbax_model/serving/...",
        "//smartass/brain/lemming/...",
        "//smartass/brain/server/...",
        "@neptune//runtime/...",
    ],
)

package(default_visibility = [
    ":internal",
])

cc_library(
    name = "optional_function_handler",
    hdrs = ["optional_function_handler.h"],
    visibility = ["//learning/infra/mira/experimental/orbax_model/tensorflow:__subpackages__"],
    deps = [
        ":interpreter_create_options",
        "//third_party/py/orbax/experimental/model/core/protos:manifest_cc_proto",
        "//third_party/py/orbax/export/protos:oex_orchestration_cc_proto",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@neptune//runtime/kernel",
    ],
)

cc_library(
    name = "optional_function_handler_factory",
    srcs = ["optional_function_handler_factory.cc"],
    hdrs = ["optional_function_handler_factory.h"],
    visibility = ["//learning/infra/mira/experimental/orbax_model/tensorflow:__subpackages__"],
    deps = [
        ":optional_function_handler",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "orbax_type_utils",
    srcs = ["orbax_type_utils.cc"],
    hdrs = ["orbax_type_utils.h"],
    deps = [
        "//third_party/py/orbax/experimental/model/core/protos:type_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/python/ifrt",
    ],
)

cc_library(
    name = "test_util",
    testonly = True,
    hdrs = ["test_util.h"],
    deps = ["//testing/base/public:gunit_for_library_testonly"],
)

cc_library(
    name = "interpreter_test_base",
    testonly = True,
    srcs = ["interpreter_test_base.cc"],
    hdrs = ["interpreter_test_base.h"],
    deps = [
        "//testing/base/public:gunit_for_library_testonly",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings:string_view",
        "@neptune//runtime/core:value",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/python/ifrt",
        "@xla//xla/python/ifrt:test_util",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:statusor",
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
    ],
    deps = [
        ":orbax_type_utils",
        "//third_party/protobuf",
        "//third_party/py/orbax/experimental/model/core/protos:manifest_cc_proto",
        "//third_party/py/orbax/experimental/model/core/protos:type_cc_proto",
        "//third_party/py/orbax/export/protos:oex_orchestration_cc_proto",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//:protobuf_lite",
        "@com_google_protobuf//util:differencer",
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
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
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
        "//third_party/py/orbax/experimental/model/core/protos:manifest_cc_proto",
        "//third_party/py/orbax/experimental/model/core/protos:type_cc_proto",
        "//third_party/py/orbax/export/protos:oex_orchestration_cc_proto",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//:protobuf_lite",
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
        "@com_google_absl//absl/time",
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
    ],
    deps = [
        ":clock",
        ":run_options",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "clock",
    hdrs = ["clock.h"],
    visibility = [
        ":internal",
        "//learning/infra/mira/experimental/orbax_model/tensorflow/kernel:__subpackages__",
        "@neptune//runtime/kernel:__subpackages__",
    ],
    deps = [
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "init_lib",
    srcs = ["init_lib.cc"],
    visibility = [":interpreter_users"],
    deps = [
        "@com_google_absl//absl/log:check",
        "@neptune//runtime/kernel",
        "@neptune//runtime/kernel:builtin_kernel",
    ],
    alwayslink = 1,
)

cc_library(
    name = "interpreter_correctness_test_lib",
    testonly = 1,
    srcs = ["interpreter_correctness_test.cc"],
    data = [
        "//learning/infra/mira/experimental/orbax_model/interpreter/testdata:stablehlo_only_models",
    ],
    visibility = [
        "//learning/infra/mira/experimental/orbax_model/interpreter/google:__subpackages__",
        "@neptune//runtime/interpreter/google:__subpackages__",
    ],
    deps = [
        ":interpreter",
        ":interpreter_create_options",
        ":interpreter_test_base",
        "//devtools/build/runtime:get_runfiles_dir",
        "//testing/base/public:gunit_for_library_testonly",
        "//testing/base/public:gunit_main",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@neptune//runtime/core:host_buffer",
        "@neptune//runtime/core:host_buffer_matcher",
        "@neptune//runtime/core:host_buffer_test_utils",
        "@neptune//runtime/core:value",
        "@neptune//runtime/core:value_matcher",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/concurrency:future",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:test_main",
    ],
)

cc_library(
    name = "interpreter_test_lib",
    testonly = 1,
    srcs = ["interpreter_test.cc"],
    copts = if_cuda(["-DGOOGLE_CUDA=1"]),
    data = [
        # The following model is to test model loading failure when there are preprocessing or
        # postprocessing functions, but no optional function handler is registered.
        "//learning/infra/mira/experimental/orbax_model/interpreter/testdata:simple_model_with_tf_processing",
        "//learning/infra/mira/experimental/orbax_model/interpreter/testdata:stablehlo_only_models",
    ],
    visibility = [
        "//learning/infra/mira/experimental/orbax_model/interpreter/google:__subpackages__",
        "@neptune//runtime/interpreter/google:__subpackages__",
    ],
    deps = [
        ":interpreter",
        ":interpreter_create_options",
        ":interpreter_test_base",
        ":optional_function_handler_factory",
        ":run_options",
        ":test_util",
        "//devtools/build/runtime:get_runfiles_dir",
        "//testing/base/public:gunit_for_library_testonly",
        "//testing/base/public:gunit_main",
        "//third_party/py/orbax/experimental/model/core/protos:manifest_cc_proto",
        "//third_party/py/orbax/export/protos:oex_orchestration_cc_proto",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@neptune//runtime/core:constants",
        "@neptune//runtime/core:flat_signature",
        "@neptune//runtime/core:host_buffer",
        "@neptune//runtime/core:host_buffer_matcher",
        "@neptune//runtime/core:host_buffer_test_utils",
        "@neptune//runtime/core:stablehlo_utils",
        "@neptune//runtime/core:value",
        "@neptune//runtime/core:value_matcher",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/ir:hlo_sharding",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/concurrency:future",
        "@xla//xla/tsl/lib/core:status_test_util",
        "@xla//xla/tsl/lib/monitoring:cell_reader",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:test_main",
    ],
)

cc_library(
    name = "interpreter_with_tf_processing_test_lib",
    testonly = 1,
    srcs = ["interpreter_with_tf_processing_test.cc"],
    data = [
        "//learning/infra/mira/experimental/orbax_model/interpreter/testdata:graph_orchestration",
        "//learning/infra/mira/experimental/orbax_model/interpreter/testdata:models_with_tf",
    ],
    visibility = [
        "//learning/infra/mira/experimental/orbax_model/interpreter/google:__subpackages__",
        "@neptune//runtime/interpreter/google:__subpackages__",
    ],
    deps = [
        ":interpreter",
        ":interpreter_test_base",
        "//devtools/build/runtime:get_runfiles_dir",
        "//learning/infra/mira/experimental/orbax_model/tensorflow:tf_compatible_interpreter_create_options",
        "//testing/base/public:gunit_for_library_testonly",
        "//testing/base/public:gunit_main",
        "//third_party/tensorflow/core:framework",
        "//third_party/tensorflow/core:protos_all_cc",
        "//third_party/tensorflow/core:testlib",
        "//third_party/tensorflow/core/framework:tensor_matcher",
        "//third_party/tensorflow/core/tfrt/runtime",
        "//third_party/tensorflow/core/tfrt/saved_model:saved_model_cpu",
        "//third_party/tensorflow/core/tfrt/saved_model:saved_model_testutil",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@neptune//runtime/core:host_buffer",
        "@neptune//runtime/core:host_buffer_matcher",
        "@neptune//runtime/core:host_buffer_test_utils",
        "@neptune//runtime/core:value",
        "@neptune//runtime/core:value_matcher",
        "@tsl//tsl/platform:tstring",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/concurrency:future",
        "@xla//xla/tsl/lib/core:status_test_util",
        "@xla//xla/tsl/platform:env",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:test_main",
    ],
)

cc_library(
    name = "interpreter_batching_with_tf_test_lib",
    testonly = 1,
    srcs = ["interpreter_batching_with_tf_test.cc"],
    data = [
        "//learning/infra/mira/experimental/orbax_model/interpreter/testdata:models_with_tf_in_batch",
    ],
    visibility = [
        "//learning/infra/mira/experimental/orbax_model/interpreter/google:__subpackages__",
        "@neptune//runtime/interpreter/google:__subpackages__",
    ],
    deps = [
        ":interpreter",
        ":interpreter_create_options",
        ":interpreter_test_base",
        "//devtools/build/runtime:get_runfiles_dir",
        "//testing/base/public:gunit_for_library_testonly",
        "//testing/base/public:gunit_main",
        "//third_party/tensorflow/core:framework",
        "//third_party/tensorflow/core:testlib",
        "//third_party/tensorflow/core/framework:tensor_matcher",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@neptune//runtime/core:host_buffer",
        "@neptune//runtime/core:host_buffer_matcher",
        "@neptune//runtime/core:host_buffer_test_utils",
        "@neptune//runtime/core:value",
        "@neptune//runtime/core:value_matcher",
        "@neptune//runtime/kernel:kernel_io_data_type",
        "@tsl//tsl/platform:tstring",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/concurrency:future",
        "@xla//xla/tsl/lib/monitoring:cell_reader",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/platform:test_main",
    ],
)

# The cc_test that runs this test library should include `TF_COST_MEASUREMENT_TYPE` and
# `TF_REQUEST_COST_ACCESSOR_TYPE` in the env variable for cost measurement-related tests. These env
# variable cannot be set in the cpp test code because it is read only once in the process. See
# `tensorflow::CreateCostMeasurements` and `tensorflow::CreateRequestCostAccessor` for more details.
cc_library(
    name = "interpreter_batching_test_lib",
    testonly = True,
    srcs = ["interpreter_batching_test.cc"],
    data = [
        "//learning/infra/mira/experimental/orbax_model/interpreter/testdata:linear_model_with_batching",
        "//learning/infra/mira/experimental/orbax_model/interpreter/testdata:linear_model_with_batching_batch_down",
        "//learning/infra/mira/experimental/orbax_model/interpreter/testdata:linear_model_with_batching_minimize_cost",
        "//learning/infra/mira/experimental/orbax_model/interpreter/testdata:linear_model_with_batching_unsharded",
    ],
    deps = [
        ":interpreter",
        ":interpreter_create_options",
        ":interpreter_test_base",
        "//devtools/build/runtime:get_runfiles_dir",
        "//testing/base/public:gunit_main",
        "//third_party/tensorflow/core/common_runtime:cost_constants",
        "//third_party/tensorflow/core/common_runtime:cost_measurement",
        "//third_party/tensorflow/core/common_runtime:cost_measurement_registry",
        "//third_party/tensorflow/core/common_runtime:request_cost",
        "//third_party/tensorflow/core/common_runtime:request_cost_accessor",
        "//third_party/tensorflow/core/common_runtime:request_cost_accessor_registry",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@neptune//runtime/core:host_buffer",
        "@neptune//runtime/core:host_buffer_matcher",
        "@neptune//runtime/core:host_buffer_test_utils",
        "@neptune//runtime/core:value",
        "@neptune//runtime/core:value_matcher",
        "@neptune//runtime/core/batching:batch_stats",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/concurrency:future",
        "@xla//xla/tsl/lib/monitoring:cell_reader",
        "@xla//xla/tsl/lib/monitoring:test_utils",
        "@xla//xla/tsl/platform:statusor",
    ],
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
        "//third_party/py/orbax/experimental/model/core/protos:manifest_cc_proto",
        "@com_google_absl//absl/base:nullability",
        "@neptune//runtime/core:allocator",
        "@neptune//runtime/core:default_allocator",
        "@neptune//runtime/core:stablehlo_optimizer",
        "@neptune//runtime/libs/ifrt/persistent_compilation_cache:ifrt_persistent_compilation_cache",
        "@tsl//tsl/platform:protobuf",
        "@xla//xla/python/ifrt",
        "@xla//xla/tsl/platform:env",
    ],
)

[cc_test(
    name = "interpreter_cpu_test" + ("_parallel_loading" if enable_parallel_loading_pipelines else "_sequential_loading"),
    args = [
        "--jsv_testonly_platform_name_override=CPU",
        "--jsv_enable_parallel_loading_pipelines=%s" % str(enable_parallel_loading_pipelines).lower(),
    ],
    env = {
        "TF_COST_MEASUREMENT_TYPE": "test",
        "TF_REQUEST_COST_ACCESSOR_TYPE": "test",
    },
    shard_count = 8,
    deps = [
        ":interpreter_batching_test_lib",
        ":interpreter_correctness_test_lib",
        ":interpreter_test_lib",
        "@neptune//runtime/core:pjrt_ifrt_cpu_client_test_lib",
    ],
) for enable_parallel_loading_pipelines in [
    True,
    False,
]]

# The TF-compatible interpreter can run models with or without TensorFlow preprocessing.
[cc_test(
    name = "tf_compatible_interpreter_cpu_test" + ("_parallel_loading" if enable_parallel_loading_pipelines else "_sequential_loading"),
    args = [
        "--jsv_testonly_platform_name_override=CPU",
        "--jsv_enable_parallel_loading_pipelines=%s" % str(enable_parallel_loading_pipelines).lower(),
    ],
    env = {
        "TF_COST_MEASUREMENT_TYPE": "test",
        "TF_REQUEST_COST_ACCESSOR_TYPE": "test",
    },
    shard_count = 8,
    deps = [
        ":interpreter_batching_test_lib",
        ":interpreter_batching_with_tf_test_lib",
        ":interpreter_correctness_test_lib",
        ":interpreter_test_lib",
        ":interpreter_with_tf_processing_test_lib",
        "//learning/infra/mira/experimental/orbax_model/tensorflow:interpreter_tensorflow_support",
        "@neptune//runtime/core:pjrt_ifrt_cpu_client_test_lib",
    ],
) for enable_parallel_loading_pipelines in [
    True,
    False,
]]

# TODO: add a dependency check test to make sure that the interpreter does not depend
# on any TensorFlow library. Currently this is blocked. See cl/704297184 for more details.
