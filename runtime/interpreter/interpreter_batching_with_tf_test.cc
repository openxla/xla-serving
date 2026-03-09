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
#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/types/span.h"
#include "third_party/neptune/runtime/core/host_buffer.h"
#include "third_party/neptune/runtime/core/host_buffer_matcher.h"
#include "third_party/neptune/runtime/core/host_buffer_test_utils.h"
#include "third_party/neptune/runtime/core/value.h"
#include "third_party/neptune/runtime/core/value_matcher.h"
#include "third_party/neptune/runtime/kernel/kernel_io_data_type.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/shape.h"
#include "third_party/tensorflow/compiler/xla/tsl/concurrency/future.h"
#include "third_party/tensorflow/compiler/xla/tsl/lib/monitoring/cell_reader.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"
#include "third_party/tensorflow/core/framework/tensor_matcher.h"
#include "third_party/tensorflow/core/framework/tensor_shape.h"
#include "third_party/tensorflow/core/framework/tensor_testutil.h"
#include "third_party/tensorflow/tsl/platform/tstring.h"

namespace jsv {
namespace {

using ::tensorflow::test::AsTensor;
using ::tensorflow::test::TensorEq;
using ::testing::HasSubstr;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;
using ::tsl::monitoring::testing::CellReader;

constexpr absl::string_view kModelPathBatchModelFuncAndPostProcessor =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/testdata/"
    "batch_model_func_and_post_processor_gen";
constexpr absl::string_view kModelPathBatchModelFuncWithScalarOutput =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/testdata/"
    "batch_model_func_with_scalar_output_gen";
constexpr absl::string_view kModelPathBatchPreProcessorAndModelFunc =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/testdata/"
    "batch_pre_processor_and_model_func_gen";
constexpr absl::string_view kModelPathBatchModelFuncWithScalarInput =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/testdata/"
    "batch_model_func_with_scalar_input_gen";
constexpr absl::string_view kModelPathBatchWholePipeline =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/testdata/"
    "batch_whole_pipeline_gen";
constexpr absl::string_view kModelPathBatchModelFuncWithScalarInputAndOutput =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/testdata/"
    "batch_model_func_with_scalar_input_and_output_gen";
constexpr absl::string_view kStringToNumberRoundTripNoBatchingModelPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/string_to_number_round_trip_no_batching_gen";
constexpr absl::string_view kStringToNumberRoundTripBatchModelFuncModelPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/string_to_number_round_trip_batch_model_func_gen";
constexpr absl::string_view
    kStringToNumberRoundTripBatchWholePipelineModelPath =
        "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
        "testdata/string_to_number_round_trip_batch_whole_pipeline_gen";
constexpr absl::string_view
    kStringToNumberRoundTripBatchPreProcessorAndModelFuncModelPath =
        "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
        "testdata/"
        "string_to_number_round_trip_batch_pre_processor_and_model_func_gen";
constexpr absl::string_view
    kStringToNumberRoundTripBatchModelFuncAndPostProcessorModelPath =
        "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
        "testdata/"
        "string_to_number_round_trip_batch_model_func_and_post_processor_gen";

class InterpreterWithTfInBatchTest : public InterpreterTestBase {
 protected:
  void SetUp() override {
    InterpreterTestBase::SetUp();

    create_options_ = std::make_unique<InterpreterCreateOptions>();
    create_options_->min_max_enqueued_batches = 250;
    const std::string platform_name_override =
        absl::GetFlag(FLAGS_jsv_testonly_platform_name_override);
    if (!platform_name_override.empty()) {
      create_options_->platform_name = platform_name_override;
    }
  }

  std::unique_ptr<InterpreterCreateOptions> create_options_;
  std::unique_ptr<Interpreter> interpreter_;
};

TEST_F(InterpreterWithTfInBatchTest, BatchModelAndPostProcessorSuccess) {
  auto executed_batch_size_reader_ = std::make_unique<CellReader<int64_t>>(
      "/orbax/serving/batching/executed_batch_size");

  create_options_->model_name = "batch_model_func_and_post_processor";
  create_options_->model_version = 0;
  create_options_->model_path = devtools_build::GetDataDependencyFilepath(
      kModelPathBatchModelFuncAndPostProcessor);
  create_options_->execute_worker_thread_pool =
      execute_worker_thread_pool_.get();
  create_options_->ifrt_client = client_;
  create_options_->computation_thread_pool = computation_thread_pool_.get();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Interpreter> interpreter,
                          Interpreter::Create(std::move(create_options_)));
  std::vector<Value> inputs = {Value(AsHostBuffer<float>(
      {{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}}, xla::ifrt::Shape({4, 2})))};
  std::vector<::testing::Matcher<const Value&>> expected_outputs = {
      ValueOfType<HostBuffer>(HostBufferEq(
          AsHostBuffer<float>({{1.0, 1.0, 5.0, 5.0, 9.0, 9.0, 13.0, 13.0}},
                              xla::ifrt::Shape({4, 2}))))};

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Value> outputs,
      interpreter
          ->RunAsync(/*pipeline_name=*/"linear_with_additional_scalar_output",
                     absl::MakeSpan(inputs))
          .Await());
  EXPECT_THAT(outputs, ElementsAreArray(expected_outputs));

  EXPECT_EQ(
      executed_batch_size_reader_->Delta("batch_model_func_and_post_processor",
                                         /*model_version=*/"0", "4"),
      1);
  for (int batch_size : {2, 8, 16}) {
    EXPECT_EQ(executed_batch_size_reader_->Delta(
                  "batch_model_func_and_post_processor",
                  /*model_version=*/"0", absl::StrCat(batch_size)),
              0);
  }
}

TEST_F(InterpreterWithTfInBatchTest, BatchModelFuncWithScalarOutputFails) {
  create_options_->model_name = "batch_model_func_with_scalar_output";
  create_options_->model_version = 0;
  create_options_->model_path = devtools_build::GetDataDependencyFilepath(
      kModelPathBatchModelFuncWithScalarOutput);
  create_options_->execute_worker_thread_pool =
      execute_worker_thread_pool_.get();
  create_options_->ifrt_client = client_;
  create_options_->computation_thread_pool = computation_thread_pool_.get();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Interpreter> interpreter,
                          Interpreter::Create(std::move(create_options_)));
  std::vector<Value> inputs = {Value(AsHostBuffer<float>(
      {{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}}, xla::ifrt::Shape({4, 2})))};

  // TODO: Update the test once we catch the error during
  // interpreter loading.
  EXPECT_THAT(
      interpreter
          ->RunAsync(/*pipeline_name=*/"linear_with_additional_scalar_output",
                     absl::MakeSpan(inputs))
          .Await(),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("Batched output buffer has 0 dimensions")));
}

TEST_F(InterpreterWithTfInBatchTest, BatchPreProcessorAndModelFuncSuccess) {
  auto executed_batch_size_reader_ = std::make_unique<CellReader<int64_t>>(
      "/orbax/serving/batching/executed_batch_size");
  create_options_->model_name = "batch_pre_processor_and_model_func";
  create_options_->model_version = 0;
  create_options_->model_path = devtools_build::GetDataDependencyFilepath(
      kModelPathBatchPreProcessorAndModelFunc);
  create_options_->execute_worker_thread_pool =
      execute_worker_thread_pool_.get();
  create_options_->ifrt_client = client_;
  create_options_->computation_thread_pool = computation_thread_pool_.get();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Interpreter> interpreter,
                          Interpreter::Create(std::move(create_options_)));
  // Since the pre-processor is in batching, we assume the inputs have been
  // converted to HostBuffer before the Interpreter::RunAsync call.
  std::vector<Value> inputs = {Value(AsHostBuffer<float>(
      {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}, xla::ifrt::Shape({4, 2})))};
  std::vector<::testing::Matcher<const Value&>> expected_outputs = {
      ValueOfType<HostBuffer>(HostBufferEq(
          AsHostBuffer<float>({{3.0, 3.0, 7.0, 7.0, 11.0, 11.0, 15.0, 15.0}},
                              xla::ifrt::Shape({4, 2}))))};

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Value> outputs,
      interpreter->RunAsync(/*pipeline_name=*/"linear", absl::MakeSpan(inputs))
          .Await());

  EXPECT_THAT(outputs, ElementsAreArray(expected_outputs));
  EXPECT_EQ(
      executed_batch_size_reader_->Delta("batch_pre_processor_and_model_func",
                                         /*model_version=*/"0", "4"),
      1);
  for (int batch_size : {2, 8, 16}) {
    EXPECT_EQ(executed_batch_size_reader_->Delta(
                  "batch_pre_processor_and_model_func",
                  /*model_version=*/"0", absl::StrCat(batch_size)),
              0);
  }
}

TEST_F(InterpreterWithTfInBatchTest, BatchModelFuncWithScalarInputFails) {
  auto executed_batch_size_reader_ = std::make_unique<CellReader<int64_t>>(
      "/orbax/serving/batching/executed_batch_size");
  create_options_->model_name = "batch_model_func_with_scalar_input";
  create_options_->model_version = 0;
  create_options_->model_path = devtools_build::GetDataDependencyFilepath(
      kModelPathBatchModelFuncWithScalarInput);
  create_options_->execute_worker_thread_pool =
      execute_worker_thread_pool_.get();
  create_options_->ifrt_client = client_;
  create_options_->computation_thread_pool = computation_thread_pool_.get();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Interpreter> interpreter,
                          Interpreter::Create(std::move(create_options_)));
  // The pre-processor exists but not in batching, so the input type is Tensor.
  std::vector<Value> inputs = {
      Value(AsTensor<float>({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0},
                            tensorflow::TensorShape({4, 2})))};

  EXPECT_THAT(
      interpreter->RunAsync(/*pipeline_name=*/"linear", absl::MakeSpan(inputs))
          .Await(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Batching input buffers must have at least one "
                         "dimension. Input at index 1 has no dimensions")));
}

TEST_F(InterpreterWithTfInBatchTest, BatchWholePipelineSuccess) {
  auto executed_batch_size_reader_ = std::make_unique<CellReader<int64_t>>(
      "/orbax/serving/batching/executed_batch_size");
  create_options_->model_name = "batch_whole_pipeline";
  create_options_->model_version = 0;
  create_options_->model_path =
      devtools_build::GetDataDependencyFilepath(kModelPathBatchWholePipeline);
  create_options_->execute_worker_thread_pool =
      execute_worker_thread_pool_.get();
  create_options_->ifrt_client = client_;
  create_options_->computation_thread_pool = computation_thread_pool_.get();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Interpreter> interpreter,
                          Interpreter::Create(std::move(create_options_)));
  // Since the pre-processor is in batching, we assume the inputs have been
  // converted to HostBuffer before the Interpreter::RunAsync call.
  std::vector<Value> inputs = {Value(AsHostBuffer<float>(
      {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}, xla::ifrt::Shape({4, 2})))};
  std::vector<::testing::Matcher<const Value&>> expected_outputs = {
      ValueOfType<HostBuffer>(HostBufferEq(
          AsHostBuffer<float>({{7.0, 7.0, 15.0, 15.0, 23.0, 23.0, 31.0, 31.0}},
                              xla::ifrt::Shape({4, 2}))))};

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Value> outputs,
      interpreter->RunAsync(/*pipeline_name=*/"linear", absl::MakeSpan(inputs))
          .Await());

  EXPECT_THAT(outputs, ElementsAreArray(expected_outputs));
  EXPECT_EQ(executed_batch_size_reader_->Delta("batch_whole_pipeline",
                                               /*model_version=*/"0", "4"),
            1);
  for (int batch_size : {2, 8, 16}) {
    EXPECT_EQ(executed_batch_size_reader_->Delta("batch_whole_pipeline",
                                                 /*model_version=*/"0",
                                                 absl::StrCat(batch_size)),
              0);
  }
}

TEST_F(InterpreterWithTfInBatchTest,
       BatchModelFuncWithScalarInputAndOutputFails) {
  auto executed_batch_size_reader_ = std::make_unique<CellReader<int64_t>>(
      "/orbax/serving/batching/executed_batch_size");
  create_options_->model_name = "batch_model_func_with_scalar_input_and_output";
  create_options_->model_version = 0;
  create_options_->model_path = devtools_build::GetDataDependencyFilepath(
      kModelPathBatchModelFuncWithScalarInputAndOutput);
  create_options_->execute_worker_thread_pool =
      execute_worker_thread_pool_.get();
  create_options_->ifrt_client = client_;
  create_options_->computation_thread_pool = computation_thread_pool_.get();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Interpreter> interpreter,
                          Interpreter::Create(std::move(create_options_)));
  // The pre-processor exists but not in batching, so the input type is Tensor.
  std::vector<Value> inputs = {
      Value(AsTensor<float>({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0},
                            tensorflow::TensorShape({4, 2})))};

  EXPECT_THAT(
      interpreter->RunAsync(/*pipeline_name=*/"linear", absl::MakeSpan(inputs))
          .Await(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Batching input buffers must have at least one "
                         "dimension. Input at index 1 has no dimensions")));
}

// Verifies that I/O data is of Tensor type when batching is disabled,
// regardless of other batching options.
TEST_F(InterpreterWithTfInBatchTest, GetModelIODataType) {
  const std::string kModelPath = devtools_build::GetDataDependencyFilepath(
      kStringToNumberRoundTripBatchWholePipelineModelPath);
  {
    auto create_options = std::make_unique<InterpreterCreateOptions>();
    create_options->enable_batching = true;
    create_options->model_name = "test_model";
    create_options->model_version = 0;
    create_options->model_path = kModelPath;
    create_options->execute_worker_thread_pool =
        execute_worker_thread_pool_.get();
    create_options->ifrt_client = client_;
    create_options->computation_thread_pool = computation_thread_pool_.get();
    TF_ASSERT_OK_AND_ASSIGN(auto interpreter,
                            Interpreter::Create(std::move(create_options)));
    EXPECT_THAT(interpreter->GetModelInputDataType("multiply_by_factor"),
                IsOkAndHolds(KernelIODataType::kHostBuffer));
    EXPECT_THAT(interpreter->GetModelOutputDataType("multiply_by_factor"),
                IsOkAndHolds(KernelIODataType::kHostBuffer));
  }
  {
    auto create_options = std::make_unique<InterpreterCreateOptions>();
    create_options->enable_batching = false;
    create_options->model_name = "test_model";
    create_options->model_version = 0;
    create_options->model_path = kModelPath;
    create_options->execute_worker_thread_pool =
        execute_worker_thread_pool_.get();
    create_options->ifrt_client = client_;
    create_options->computation_thread_pool = computation_thread_pool_.get();
    TF_ASSERT_OK_AND_ASSIGN(auto interpreter,
                            Interpreter::Create(std::move(create_options)));
    EXPECT_THAT(interpreter->GetModelInputDataType("multiply_by_factor"),
                IsOkAndHolds(KernelIODataType::kTensor));
    EXPECT_THAT(interpreter->GetModelOutputDataType("multiply_by_factor"),
                IsOkAndHolds(KernelIODataType::kTensor));
  }
}

class InterpreterBatchComponentTest
    : public InterpreterWithTfInBatchTest,
      public ::testing::WithParamInterface<absl::string_view> {
 protected:
  void SetUp() override {
    InterpreterWithTfInBatchTest::SetUp();
    const absl::string_view model_path = GetParam();
    create_options_->model_name = "batch_with_tf_test_model";
    create_options_->model_version = 0;
    create_options_->model_path =
        devtools_build::GetDataDependencyFilepath(model_path);
    create_options_->execute_worker_thread_pool =
        execute_worker_thread_pool_.get();
    create_options_->ifrt_client = client_;
    create_options_->computation_thread_pool = computation_thread_pool_.get();
    TF_ASSERT_OK_AND_ASSIGN(interpreter_,
                            Interpreter::Create(std::move(create_options_)));

    // Assume the input and output are converted to correct types according to
    // the interpreter's input and output data types.
    TF_ASSERT_OK_AND_ASSIGN(
        KernelIODataType input_data_type,
        interpreter_->GetModelInputDataType(pipeline_name_));
    TF_ASSERT_OK_AND_ASSIGN(
        KernelIODataType output_data_type,
        interpreter_->GetModelOutputDataType(pipeline_name_));
    if (input_data_type == KernelIODataType::kTensor) {
      input_1_ = {Value(AsTensor<tsl::tstring>(input_1_values_,
                                               tensorflow::TensorShape({4})))};
      input_2_ = {Value(AsTensor<tsl::tstring>(input_2_values_,
                                               tensorflow::TensorShape({8})))};
    } else {
      input_1_ = {Value(
          AsHostBuffer<tsl::tstring>(input_1_values_, xla::ifrt::Shape({4})))};
      input_2_ = {Value(
          AsHostBuffer<tsl::tstring>(input_2_values_, xla::ifrt::Shape({8})))};
    }
    if (output_data_type == KernelIODataType::kTensor) {
      expected_output_1_ = {
          {ValueOfType<tensorflow::Tensor>(TensorEq(AsTensor<tsl::tstring>(
              expected_output_1_values_, tensorflow::TensorShape({4}))))}};
      expected_output_2_ = {
          {ValueOfType<tensorflow::Tensor>(TensorEq(AsTensor<tsl::tstring>(
              expected_output_2_values_, tensorflow::TensorShape({8}))))}};
    } else {
      expected_output_1_ = {
          {ValueOfType<HostBuffer>(HostBufferEq(AsHostBuffer<tsl::tstring>(
              expected_output_1_values_, xla::ifrt::Shape({4}))))}};
      expected_output_2_ = {
          {ValueOfType<HostBuffer>(HostBufferEq(AsHostBuffer<tsl::tstring>(
              expected_output_2_values_, xla::ifrt::Shape({8}))))}};
    }

    if (model_path == kStringToNumberRoundTripNoBatchingModelPath) {
      batch_enabled_ = false;
    }
  };

  std::vector<tsl::tstring> input_1_values_ = {"1.0", "2.0", "3.0", "4.0"};
  std::vector<tsl::tstring> input_2_values_ = {"5.0", "6.0",  "7.0",  "8.0",
                                               "9.0", "10.0", "11.0", "12.0"};
  std::vector<tsl::tstring> expected_output_1_values_ = {"2", "4", "6", "8"};
  std::vector<tsl::tstring> expected_output_2_values_ = {
      "10", "12", "14", "16", "18", "20", "22", "24"};
  std::vector<Value> input_1_;
  std::vector<Value> input_2_;
  std::vector<::testing::Matcher<const Value&>> expected_output_1_;
  std::vector<::testing::Matcher<const Value&>> expected_output_2_;
  bool batch_enabled_ = true;
  absl::string_view pipeline_name_ = "multiply_by_factor";
};

TEST_P(InterpreterBatchComponentTest, StringToNumberRoundTripSuccess) {
  auto executed_batch_size_reader_ = std::make_unique<CellReader<int64_t>>(
      "/orbax/serving/batching/executed_batch_size");
  int request_count = 24;
  std::vector<tsl::Future<std::vector<Value>>> futures;

  for (int i = 0; i < request_count / 2; ++i) {
    futures.push_back(
        interpreter_->RunAsync(pipeline_name_, absl::MakeSpan(input_1_)));
    futures.push_back(
        interpreter_->RunAsync(pipeline_name_, absl::MakeSpan(input_2_)));
  }
  for (int i = 0; i < futures.size(); ++i) {
    TF_ASSERT_OK_AND_ASSIGN(std::vector<Value> output_1, futures[i].Await());
    EXPECT_THAT(output_1, ElementsAreArray(expected_output_1_));
    TF_ASSERT_OK_AND_ASSIGN(std::vector<Value> output_2, futures[++i].Await());
    EXPECT_THAT(output_2, ElementsAreArray(expected_output_2_));
  }
  // The allowed batch sizes are 2, 4, 8, 16, 32. There are 24 requests with
  // batch sizes of 4 and 8. The total batch size is 144. With large batch
  // splitting enabled, there will be 4 batch of size 32 and 1 batch of size 16.
  if (batch_enabled_) {
    EXPECT_EQ(executed_batch_size_reader_->Delta("batch_with_tf_test_model",
                                                 /*model_version=*/"0", "32"),
              4);
    EXPECT_EQ(executed_batch_size_reader_->Delta("batch_with_tf_test_model",
                                                 /*model_version=*/"0", "16"),
              1);
    for (int batch_size : {2, 4, 8}) {
      EXPECT_EQ(executed_batch_size_reader_->Delta("batch_with_tf_test_model",
                                                   /*model_version=*/"0",
                                                   absl::StrCat(batch_size)),
                0);
    }
  } else {
    for (int batch_size : {2, 4, 8, 16, 32}) {
      EXPECT_EQ(executed_batch_size_reader_->Delta("batch_with_tf_test_model",
                                                   /*model_version=*/"0",
                                                   absl::StrCat(batch_size)),
                0);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    InterpreterBatchComponentSameInputAndOutputTest,
    InterpreterBatchComponentTest,
    ::testing::Values(
        kStringToNumberRoundTripNoBatchingModelPath,
        kStringToNumberRoundTripBatchModelFuncModelPath,
        kStringToNumberRoundTripBatchWholePipelineModelPath,
        kStringToNumberRoundTripBatchPreProcessorAndModelFuncModelPath,
        kStringToNumberRoundTripBatchModelFuncAndPostProcessorModelPath));

}  // namespace
}  // namespace jsv
