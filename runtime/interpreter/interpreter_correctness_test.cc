#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "devtools/build/runtime/get_runfiles_dir.h"
#include "third_party/neptune/runtime/interpreter/interpreter.h"
#include "third_party/neptune/runtime/interpreter/interpreter_create_options.h"
#include "third_party/neptune/runtime/interpreter/interpreter_test_base.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/flags/flag.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/types/span.h"
#include "third_party/neptune/runtime/core/host_buffer.h"
#include "third_party/neptune/runtime/core/host_buffer_matcher.h"
#include "third_party/neptune/runtime/core/host_buffer_test_utils.h"
#include "third_party/neptune/runtime/core/value.h"
#include "third_party/neptune/runtime/core/value_matcher.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/shape.h"
#include "third_party/tensorflow/compiler/xla/tsl/concurrency/future.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"

namespace jsv {
namespace {
using ::testing::ElementsAreArray;

constexpr absl::string_view kPassThroughModelPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/pass_through_gen";
constexpr absl::string_view kXTimesYModelPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/x_times_y_gen";
constexpr absl::string_view kLinearModelPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/linear_model_gen";
constexpr absl::string_view kMakeNumberFromDigitsModelPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/make_number_from_digits_gen";
constexpr absl::string_view kDynamicShapePath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/dynamic_shape_gen";
constexpr absl::string_view kLinearModelPolymorphicPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/linear_model_polymorphic_gen";
constexpr absl::string_view kLinearModelPolymorphicConcretizedPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/linear_model_with_polymorphic_input_concretized_gen";
constexpr absl::string_view kLinearModelWithUnusedInputPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/linear_model_with_unused_input_gen";
constexpr absl::string_view kLinearModelWithUnusedParamPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/linear_model_with_unused_param_gen";
constexpr absl::string_view kLinearModelWithUnusedInputAndParamPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/linear_model_with_unused_input_and_param_gen";
constexpr absl::string_view kLinearModelUnshardedPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/linear_model_unsharded_gen";
constexpr absl::string_view kLinearModelWithAutoLayoutPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/linear_model_with_auto_layout_gen";
constexpr absl::string_view kMultipleApplyFnSharingCkptPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/multiple_apply_fn_sharing_ckpt_gen";
constexpr absl::string_view kMultipleApplyFnSharingCkptWithPolymorphicPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/multiple_apply_fn_sharing_ckpt_with_polymorphic_gen";
constexpr absl::string_view kMatrixMultiplicationPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/matrix_multiplication_gen";

class InterpreterMatchOutputTest
    : public InterpreterTestBase,
      public testing::WithParamInterface<testing::tuple<
          InterpreterMatchOutputTestParam, /*enable_async_run=*/bool,
          /*lazy_loading_pipelines=*/bool>> {
 protected:
  void SetUp() override {
    InterpreterTestBase::SetUp();
    const auto& [params, unused, lazy_loading_pipelines] = GetParam();
    auto create_options = std::make_unique<InterpreterCreateOptions>();
    const std::string platform_name_override =
        absl::GetFlag(FLAGS_jsv_testonly_platform_name_override);

    // Explicit Hardware constraint validation:
    // The GPU CI environment (requires-gpu-nvidia:2) only provides 2 devices.
    // Skip tests that require a larger device topology.
    if (platform_name_override == "cuda" && !params.is_gpu_test_supported) {
      GTEST_SKIP() << "Test strictly requires 4+ hardware devices. Skipping "
                      "on GPU CI which is constrained to 2x NVIDIA GPUs."
                   << params.testcase_name;
    }

    if (!platform_name_override.empty()) {
      create_options->platform_name = platform_name_override;
    }
    create_options->lazy_loading_pipelines = lazy_loading_pipelines;
    create_options->model_name = params.model_name;
    create_options->model_version = 0;
    create_options->model_path =
        devtools_build::GetDataDependencyFilepath(params.model_path);
    create_options->execute_worker_thread_pool =
        execute_worker_thread_pool_.get();
    create_options->ifrt_client = client_;
    create_options->computation_thread_pool = computation_thread_pool_.get();
    TF_ASSERT_OK_AND_ASSIGN(interpreter_,
                            Interpreter::Create(std::move(create_options)));
  }

  std::unique_ptr<Interpreter> interpreter_;
};

std::vector<InterpreterMatchOutputTestParam> GetPassThroughTestParams() {
  // Test a pass through model.
  return {
      {
          // No weights
          .testcase_name = "pass_through_model_no_weights",
          .model_path = kPassThroughModelPath,
          .pipeline_name = "pass_through",
          .inputs = {Value(
              AsHostBuffer<float>({{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}},
                                  xla::ifrt::Shape({4, 2})))},
          .expected_outputs = {ValueOfType<HostBuffer>(HostBufferEq(
              AsHostBuffer<float>({{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}},
                                  xla::ifrt::Shape({4, 2}))))},
      },
  };
}

std::vector<InterpreterMatchOutputTestParam> GetXTimesYTestParams() {
  // Test a model that does x * y.
  return {
      {
          // No checkpoint and no SPMD.
          .testcase_name = "x_times_y_model_no_weights_no_spmd",
          .model_path = kXTimesYModelPath,
          .pipeline_name = "x_times_y",
          .inputs = {Value(AsHostBuffer<float>(
                         {{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}},
                         xla::ifrt::Shape({4, 2}))),
                     Value(AsHostBuffer<float>({{10.0, 11.0, 12.0, 13.0}},
                                               xla::ifrt::Shape({2, 2})))},

          .expected_outputs = {ValueOfType<HostBuffer>(
              HostBufferEq(AsHostBuffer<float>(
                  {{34.0, 37.0, 78.0, 85.0, 122.0, 133.0, 166.0, 181.0}},
                  xla::ifrt::Shape({4, 2}))))},
      },
  };
}

std::vector<InterpreterMatchOutputTestParam> GetLinearModelTestParams() {
  // Tests linear model with weights and bias. This model
  // implements a simple linear transformation: y = x * w + b
  // weights are
  // Array([[1., 1.],
  //        [1., 1.]], dtype=float32)
  // bias are
  // Array([1, 2], dtype=float32)
  return {
      {
          // Basic linear model.
          .testcase_name = "linear_model",
          .model_path = kLinearModelPath,
          .pipeline_name = "linear",
          .inputs = {Value(
              AsHostBuffer<float>({{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}},
                                  xla::ifrt::Shape({4, 2})))},
          .expected_outputs = {ValueOfType<HostBuffer>(
              HostBufferEq((AsHostBuffer<float>(
                  {{2.0, 3.0, 6.0, 7.0, 10.0, 11.0, 14.0, 15.0}},
                  xla::ifrt::Shape({4, 2})))))},
      },
      {
          // Use the signature key "serving_default" that
          // is an alias of pipeline name "linear".
          .testcase_name = "linear_model_serving_default",
          .model_path = kLinearModelPath,
          .pipeline_name = "serving_default",
          .inputs = {Value(
              AsHostBuffer<float>({{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}},
                                  xla::ifrt::Shape({4, 2})))},
          .expected_outputs = {ValueOfType<HostBuffer>(
              HostBufferEq((AsHostBuffer<float>(
                  {{2.0, 3.0, 6.0, 7.0, 10.0, 11.0, 14.0, 15.0}},
                  xla::ifrt::Shape({4, 2})))))},
      },
      {
          // Unsharded.
          .testcase_name = "linear_model_unsharded",
          .model_path = kLinearModelUnshardedPath,
          .pipeline_name = "linear",
          .inputs = {Value(
              AsHostBuffer<float>({{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}},
                                  xla::ifrt::Shape({4, 2})))},
          .expected_outputs = {ValueOfType<HostBuffer>(
              HostBufferEq((AsHostBuffer<float>({{2.0, 3.0, 6.0, 7.0, 10.0,
                                                  11.0, 14.0, 15.0}},
                                                xla::ifrt::Shape({4, 2})))))},
          .is_gpu_test_supported = true,
      },
  };
}

std::vector<InterpreterMatchOutputTestParam>
GetLinearModelPolymorphicTestParams() {
  // Tests linear model with weights and bias. This model
  // implements a simple linear transformation: y = x * w + b
  // weights are
  // Array([[1., 1.],
  //        [1., 1.]], dtype=float32)
  // bias are
  // Array([1, 2], dtype=float32)
  return {
      {
          // Test polymorphic batch size. Here the batch size is 2.
          .testcase_name = "linear_model_poly_input_batch_2",
          .model_path = kLinearModelPolymorphicPath,
          .pipeline_name = "linear",
          .inputs = {Value(AsHostBuffer<float>({{5.0, 9.0, 10.0, 11.0}},
                                               xla::ifrt::Shape({2, 2})))},
          .expected_outputs = {ValueOfType<HostBuffer>(
              HostBufferEq((AsHostBuffer<float>({{15.0, 16.0, 22.0, 23.0}},
                                                xla::ifrt::Shape({2, 2})))))},
      },
  };
}

std::vector<InterpreterMatchOutputTestParam>
GetLinearModelUnusedInputAndParamTestParams() {
  // Tests linear model with weights and bias and unused input and/or param.
  // This model implements a simple linear transformation: y = x * w + b
  // weights are
  // Array([[1., 1.],
  //        [1., 1.]], dtype=float32)
  // bias are
  // Array([1, 2], dtype=float32)
  return {
      {
          // Test with unused input.
          .testcase_name = "linear_model_unused_input",
          .model_path = kLinearModelWithUnusedInputPath,
          .pipeline_name = "linear",
          .inputs = {/*unused_input=*/Value(AsHostBuffer<float>(
                         {{-1.0, -1.0}}, xla::ifrt::Shape({2, 1}))),
                     Value(AsHostBuffer<float>({{0.0, 1.0, 2.0, 3.0}},
                                               xla::ifrt::Shape({2, 2})))},
          .expected_outputs = {ValueOfType<HostBuffer>(
              HostBufferEq((AsHostBuffer<float>({{2.0, 3.0, 6.0, 7.0}},
                                                xla::ifrt::Shape({2, 2})))))},
      },
      {
          // Test with unused param.
          .testcase_name = "linear_model_unused_param",
          .model_path = kLinearModelWithUnusedParamPath,
          .pipeline_name = "linear",
          .inputs = {Value(AsHostBuffer<float>({{4.0, 5.0, 6.0, 7.0}},
                                               xla::ifrt::Shape({2, 2})))},
          .expected_outputs = {ValueOfType<HostBuffer>(
              HostBufferEq((AsHostBuffer<float>({{10.0, 11.0, 14.0, 15.0}},
                                                xla::ifrt::Shape({2, 2})))))},
      },
      {
          // Test with unused input and param.
          .testcase_name = "linear_model_unused_input_and_param",
          .model_path = kLinearModelWithUnusedInputAndParamPath,
          .pipeline_name = "linear",
          .inputs = {/*unused_input=*/Value(AsHostBuffer<float>(
                         {{-1.0, -1.0}}, xla::ifrt::Shape({2, 1}))),
                     Value(AsHostBuffer<float>({{0.0, 1.0, 2.0, 3.0}},
                                               xla::ifrt::Shape({2, 2})))},
          .expected_outputs = {ValueOfType<HostBuffer>(
              HostBufferEq((AsHostBuffer<float>({{2.0, 3.0, 6.0, 7.0}},
                                                xla::ifrt::Shape({2, 2})))))},
      },
  };
}

std::vector<InterpreterMatchOutputTestParam>
GetLinearModelPolymorphicConcretizedTestParams() {
  // Tests simple linear model with polymorphic input that is
  // concretized and exported to a set of StableHLOs. The linear
  // transformation is:
  //
  // When batch_size <= 2: y = x * w + b.
  // When 2 < batch_size <= 4: y = x * w.
  // When batch_size > 4: y = x.
  //
  // weights are
  // Array([[1., 1.],
  //        [1., 1.]], dtype=float32)
  // bias are
  // Array([1, 1], dtype=float32)
  return {
      {
          .testcase_name = "linear_model_poly_concretized_input_batch_2",
          .model_path = kLinearModelPolymorphicConcretizedPath,
          .pipeline_name = "linear_with_conditional_bias",
          .inputs = {Value(AsHostBuffer<float>({{0.0, 1.0, 2.0, 3.0}},
                                               xla::ifrt::Shape({2, 2})))},
          .expected_outputs = {ValueOfType<HostBuffer>(
              HostBufferEq((AsHostBuffer<float>({{2.0, 2.0, 6.0, 6.0}},
                                                xla::ifrt::Shape({2, 2})))))},
      },
      {
          .testcase_name = "linear_model_poly_concretized_input_batch_4",
          .model_path = kLinearModelPolymorphicConcretizedPath,
          .pipeline_name = "linear_with_conditional_bias",
          .inputs = {Value(
              AsHostBuffer<float>({{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}},
                                  xla::ifrt::Shape({4, 2})))},
          .expected_outputs = {ValueOfType<HostBuffer>(HostBufferEq(
              (AsHostBuffer<float>({{1.0, 1.0, 5.0, 5.0, 9.0, 9.0, 13.0, 13.0}},
                                   xla::ifrt::Shape({4, 2})))))},
      },
      {
          .testcase_name = "linear_model_poly_concretized_input_batch_8",
          .model_path = kLinearModelPolymorphicConcretizedPath,
          .pipeline_name = "linear_with_conditional_bias",
          .inputs = {Value(
              AsHostBuffer<float>({{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                                    9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0}},
                                  xla::ifrt::Shape({8, 2})))},
          .expected_outputs = {ValueOfType<HostBuffer>(HostBufferEq((
              AsHostBuffer<float>({{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                                    9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0}},
                                  xla::ifrt::Shape({8, 2})))))},
      },
  };
}

// Helper function to generate test parameters for
// kMakeNumberFromDigitsModelPath
std::vector<InterpreterMatchOutputTestParam>
GetMakeNumberFromDigitsTestParams() {
  // Tests a model that combines digits into a number:
  //   y1 = first_digit * 10000 + second_digit * 1000 +
  //       third_digit * 100 + fourth_digit * 10 + fifth_digit
  //   y2 = first_digit * 0.1 + second_digit * 0.01 +
  //       third_digit * 0.001 + fourth_digit * 0.0001 +
  //       fifth_digit * 0.00001
  // where:
  //   - first_digit: First digit of the number with shape
  //   (input_dim). This is from the checkpoint. The value is [1, 2].
  //   - second_digit: Second digit of the number with shape
  //   (input_dim). This is from the checkpoint. The value is [3, 4].
  //   - third_digit, fourth_digit, fifth_digit: digit of the number
  //   with shape (batch_size, input_dim).
  return {
      {
          .testcase_name = "make_number_from_digits_model",
          .model_path = kMakeNumberFromDigitsModelPath,
          .pipeline_name = "make_number_from_digits",
          .inputs =
              {/*third_digit=*/
               Value(AsHostBuffer<int32_t>({4, 5, 6, 7, 8, 9, 3, 2},
                                           xla::ifrt::Shape({4, 2}))),
               /*fourth_digit=*/
               Value(AsHostBuffer<int32_t>({1, 2, 3, 4, 5, 6, 7, 8},
                                           xla::ifrt::Shape({4, 2}))),
               /*fifth_digit=*/
               Value(AsHostBuffer<int32_t>({9, 8, 7, 6, 5, 4, 3, 2},
                                           xla::ifrt::Shape({4, 2})))},
          .expected_outputs =
              {ValueOfType<HostBuffer>(HostBufferEq(AsHostBuffer<int32_t>(
                   {13419, 24528, 13637, 24746, 13855, 24964, 13373, 24282},
                   xla::ifrt::Shape({4, 2})))),
               ValueOfType<HostBuffer>(HostBufferEq(
                   AsHostBuffer<float>({{0.13419, 0.24528, 0.13637, 0.24746,
                                         0.13855, 0.24964, 0.13373, 0.24282}},
                                       xla::ifrt::Shape({4, 2}))))},
      },
  };
}

std::vector<InterpreterMatchOutputTestParam> GetDynamicShapeTestParams() {
  // This model is a tiny DNN model with input that has dynamic shape:
  //   result = x * params['one'] * params['two']
  // where
  //   - input x has shape (batch_size, 1024) and the batch size is unknown.
  //   - params['one'] has shape (1024, 256).
  //   - params['two'] has shape (256, 1).
  //   - result has shape (batch_size, 1).
  return {
      {
          .testcase_name = "dynamic_shape_model",
          .model_path = kDynamicShapePath,
          .pipeline_name = "tiny_dnn",
          .inputs = {Value(AsHostBuffer<float>(std::vector<float>(2048, 2.0),
                                               xla::ifrt::Shape({2, 1024})))},
          .expected_outputs = {ValueOfType<HostBuffer>(
              HostBufferEq(AsHostBuffer<float>({524288.0, 524288.0},
                                               xla::ifrt::Shape({2, 1}))))},
      },
  };
}

std::vector<InterpreterMatchOutputTestParam>
GetMultipleApplyFnSharingCkptTestParams() {
  // This model has two `apply_fn`s that share the same checkpoint.
  // The first apply_fn is:
  //   y1 = base + x
  // The second apply_fn is:
  //   y2 = base - x
  // where:
  //   - x: Input data with shape (batch_size, input_dim).
  //   - base: Weight with shape (input_dim), initialized with ones.
  //   - y1: Output data with shape (batch_size, input_dim).
  //   - y2: Output data with shape (batch_size, input_dim).
  // Example:
  //   x = [[1, 2], [3, 4]]
  //   By default
  //     weights is [1, 1]
  //   Therefore,
  //     y1 = [[2, 3], [4, 5]]
  //     y2 = [[0, -1], [-2, -3]]
  return {
      {
          .testcase_name = "multi_pipeline_model_run_plus",
          .model_path = kMultipleApplyFnSharingCkptPath,
          .pipeline_name = "plus_sig",
          .inputs = {Value(
              AsHostBuffer<float>({{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}},
                                  xla::ifrt::Shape({4, 2})))},
          .expected_outputs = {ValueOfType<HostBuffer>(HostBufferEq(
              AsHostBuffer<float>({{2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}},
                                  xla::ifrt::Shape({4, 2}))))},
      },
      {
          .testcase_name = "multi_pipeline_model_run_minus",
          .model_path = kMultipleApplyFnSharingCkptPath,
          .pipeline_name = "minus_sig",
          .inputs = {Value(
              AsHostBuffer<float>({{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}},
                                  xla::ifrt::Shape({4, 2})))},
          .expected_outputs = {ValueOfType<HostBuffer>(
              HostBufferEq(AsHostBuffer<float>(
                  {{0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0}},
                  xla::ifrt::Shape({4, 2}))))},
      },
      {
          // Use polymorphic version of the model.
          // is not the leading dimension.
          .testcase_name = "multi_pipeline_poly_model_run_plus",
          .model_path = kMultipleApplyFnSharingCkptWithPolymorphicPath,
          .pipeline_name = "plus_sig",
          .inputs = {Value(
              AsHostBuffer<float>({{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}},
                                  xla::ifrt::Shape({4, 2})))},
          .expected_outputs = {ValueOfType<HostBuffer>(HostBufferEq(
              AsHostBuffer<float>({{2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}},
                                  xla::ifrt::Shape({4, 2}))))},
      },
      {
          // Use polymorphic version of the model.
          .testcase_name = "multi_pipeline_poly_model_run_minus2",
          .model_path = kMultipleApplyFnSharingCkptWithPolymorphicPath,
          .pipeline_name = "minus2_sig",
          .inputs = {Value(
              AsHostBuffer<float>({{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}},
                                  xla::ifrt::Shape({4, 2})))},
          .expected_outputs = {ValueOfType<HostBuffer>(
              HostBufferEq(AsHostBuffer<float>({{1.0, 0.0, -1.0, -2.0, -3.0,
                                                 -4.0, -5.0, -6.0}},
                                               xla::ifrt::Shape({4, 2}))))},
      },
  };
}

std::vector<InterpreterMatchOutputTestParam>
GetMatrixMultiplicationTestParams() {
  // This model is a matrix multiplication model with replicated weights:
  //   result = x * params['weights']
  // where
  //   - input x has shape (batch_size, input_dim).
  //   - params['weights'] has shape (input_dim, output_dim).
  //   - result has shape (batch_size, output_dim).
  return {
      {
          .testcase_name = "matrix_multiplication_model",
          .model_path = kMatrixMultiplicationPath,
          .pipeline_name = "serving_default",
          .inputs = {Value(AsHostBuffer<float>({{1.0, 2.0, 3.0, 4.0}},
                                               xla::ifrt::Shape({2, 2})))},
          .expected_outputs = {ValueOfType<HostBuffer>(
              HostBufferEq(AsHostBuffer<float>({{3.0, 3.0, 7.0, 7.0}},
                                               xla::ifrt::Shape({2, 2}))))},
      },
  };
}

void AddTestParams(absl::Span<const InterpreterMatchOutputTestParam> param,
                   std::vector<InterpreterMatchOutputTestParam>& all_params) {
  all_params.insert(all_params.end(), param.begin(), param.end());
}

// Function to combine all the test parameters into a single vector.
std::vector<InterpreterMatchOutputTestParam> GetAllTestParams() {
  std::vector<InterpreterMatchOutputTestParam> all_params;
  AddTestParams(GetPassThroughTestParams(), all_params);
  AddTestParams(GetXTimesYTestParams(), all_params);
  AddTestParams(GetLinearModelTestParams(), all_params);
  AddTestParams(GetLinearModelPolymorphicTestParams(), all_params);
  AddTestParams(GetLinearModelUnusedInputAndParamTestParams(), all_params);
  AddTestParams(GetLinearModelPolymorphicConcretizedTestParams(), all_params);
  AddTestParams(GetMakeNumberFromDigitsTestParams(), all_params);
  AddTestParams(GetDynamicShapeTestParams(), all_params);
  AddTestParams(GetMultipleApplyFnSharingCkptTestParams(), all_params);
  AddTestParams(GetMatrixMultiplicationTestParams(), all_params);
  return all_params;
}

TEST_P(InterpreterMatchOutputTest, LoadAndRunSucceeds) {
  const auto& [params, enable_async_run, unused] = GetParam();
  std::vector<Value> outputs;
  if (!enable_async_run) {
    TF_ASSERT_OK_AND_ASSIGN(
        outputs,
        interpreter_->Run(params.pipeline_name, absl::MakeSpan(params.inputs)));
  } else {
    tsl::Future<std::vector<Value>> outputs_future = interpreter_->RunAsync(
        params.pipeline_name, absl::MakeSpan(params.inputs));
    TF_ASSERT_OK_AND_ASSIGN(outputs, outputs_future.Await());
  }

  EXPECT_THAT(outputs, ElementsAreArray(params.expected_outputs));
}

std::vector<InterpreterMatchOutputTestParam>
GetLinearModelWithAutoLayoutTestParams() {
  // Tests linear model with weights and bias. This model
  // implements a simple linear transformation: y = x * w + b
  // weights are
  // Array([[1., 1.],
  //        [1., 1.]], dtype=float32)
  // bias are
  // Array([1, 2], dtype=float32)
  return {
      {
          // Basic linear model.
          .testcase_name = "linear_model_with_auto_layout",
          .model_path = kLinearModelWithAutoLayoutPath,
          .pipeline_name = "linear",
          .inputs = {Value(
              AsHostBuffer<float>({{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}},
                                  xla::ifrt::Shape({4, 2})))},
          .expected_outputs = {ValueOfType<HostBuffer>(
              HostBufferEq((AsHostBuffer<float>(
                  {{2.0, 3.0, 6.0, 7.0, 10.0, 11.0, 14.0, 15.0}},
                  xla::ifrt::Shape({4, 2})))))},
      },
  };
}

INSTANTIATE_TEST_SUITE_P(
    InterpreterMatchOutputTestInstances, InterpreterMatchOutputTest,
    testing::Combine(testing::ValuesIn(GetAllTestParams()),
                     /*enable_async_run=*/testing::Bool(),
                     /*lazy_loading_pipelines=*/testing::Bool()),
    [](const testing::TestParamInfo<InterpreterMatchOutputTest::ParamType>&
           info) {
      const InterpreterMatchOutputTestParam& param = std::get<0>(info.param);
      return absl::StrCat(param.testcase_name, "_",
                          std::get<1>(info.param) ? "async" : "sync",
                          std::get<2>(info.param) ? "_lazy_loading" : "");
    });

INSTANTIATE_TEST_SUITE_P(
    InterpreterMatchOutputWithLayoutTestInstances, InterpreterMatchOutputTest,
    testing::Combine(
        testing::ValuesIn(GetLinearModelWithAutoLayoutTestParams()),
        /*enable_async_run=*/testing::Bool(),
        /*lazy_loading_pipelines=*/testing::Bool()),
    [](const testing::TestParamInfo<InterpreterMatchOutputTest::ParamType>&
           info) {
      const InterpreterMatchOutputTestParam& param = std::get<0>(info.param);
      return absl::StrCat(param.testcase_name, "_",
                          std::get<1>(info.param) ? "async" : "sync",
                          std::get<2>(info.param) ? "_lazy_loading" : "");
    });

}  // namespace
}  // namespace jsv
