#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "devtools/build/runtime/get_runfiles_dir.h"
#include "third_party/neptune/runtime/interpreter/interpreter.h"
#include "third_party/neptune/runtime/interpreter/interpreter_test_base.h"
#include "learning/infra/mira/experimental/orbax_model/tensorflow/tf_compatible_interpreter_create_options.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/flags/declare.h"
#include "third_party/absl/flags/flag.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
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
#include "third_party/tensorflow/compiler/xla/tsl/lib/core/status_test_util.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/env.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/threadpool.h"
#include "third_party/tensorflow/compiler/xla/xla_data.proto.h"
#include "third_party/tensorflow/core/framework/tensor.h"
#include "third_party/tensorflow/core/framework/tensor_matcher.h"
#include "third_party/tensorflow/core/framework/tensor_shape.h"
#include "third_party/tensorflow/core/framework/tensor_testutil.h"
#include "third_party/tensorflow/core/framework/types.proto.h"
#include "third_party/tensorflow/core/tfrt/runtime/runtime.h"
#include "third_party/tensorflow/core/tfrt/saved_model/saved_model.h"
#include "third_party/tensorflow/core/tfrt/saved_model/saved_model_testutil.h"
#include "third_party/tensorflow/tsl/platform/tstring.h"

ABSL_DECLARE_FLAG(std::string, jsv_testonly_platform_name_override);

namespace jsv {
namespace {
using ::tensorflow::test::AsTensor;
using ::tensorflow::test::TensorEq;
using ::testing::Each;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Pair;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

constexpr absl::string_view kSimpleModelWithTfProcessingPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/simple_model_with_tf_processing_gen";
constexpr absl::string_view kSimpleModelWithTfProcessingUnshardedPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/testdata/"
    "simple_model_with_tf_processing_unsharded_gen";
constexpr absl::string_view kSimpleModelWithTfPreprocessorOnlyPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/simple_model_with_tf_preprocessor_only_gen";
constexpr absl::string_view kSimpleModelWithTfResourcesPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/simple_model_with_tf_resources_gen";
constexpr absl::string_view kMnistModelPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/mnist_gen";
constexpr absl::string_view kIdentityMultiIoModelUseTuplePath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/identity_multi_io_model_use_tuple_gen";
constexpr absl::string_view kIdentityMultiIoModelUseDictPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/identity_multi_io_model_use_dict_gen";
constexpr absl::string_view kIdentityMultiIoModelUseDictUnshardedPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/identity_multi_io_model_use_dict_unsharded_gen";
constexpr absl::string_view kIdentityMultiIoModelKwargsOnlyPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/identity_multi_io_model_kwargs_only_gen";
constexpr absl::string_view kIdentityMultiIoModelKwargsOnlyUnshardedPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/identity_multi_io_model_kwargs_only_unsharded_gen";
constexpr absl::string_view kStringToNumberRoundTripModelPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/string_to_number_round_trip_batch_model_func_gen";
constexpr absl::string_view kGraphOrchestrationModelPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/graph_orchestration_gen";

class InterpreterWithTfProcessingTestBase : public InterpreterTestBase {
 protected:
  void SetUp() override {
    InterpreterTestBase::SetUp();

    tfrt_runtime_ = tensorflow::tfrt_stub::DefaultTfrtRuntime(8);
    create_options_ = std::make_unique<TfCompatibleInterpreterCreateOptions>();
    create_options_->tfrt_options.saved_model_options =
        tensorflow::tfrt_stub::DefaultSavedModelOptions(tfrt_runtime_.get());
    const std::string platform_name_override =
        absl::GetFlag(FLAGS_jsv_testonly_platform_name_override);
    if (!platform_name_override.empty()) {
      create_options_->platform_name = platform_name_override;
    }
  }

  std::unique_ptr<tensorflow::tfrt_stub::Runtime> tfrt_runtime_;
  std::unique_ptr<TfCompatibleInterpreterCreateOptions> create_options_;
  // The interpreter should be destroyed first.
  std::unique_ptr<Interpreter> interpreter_;
};

class InterpreterWithTfProcessingTest
    : public InterpreterWithTfProcessingTestBase {};

// Parameterized test to load and run the model and check the outputs.
class InterpreterWithTfProcessingMatchOutputTest
    : public InterpreterWithTfProcessingTest,
      public testing::WithParamInterface<testing::tuple<
          InterpreterMatchOutputTestParam, /*enable_async_run=*/bool,
          /*lazy_loading_pipelines=*/bool>> {
 protected:
  void SetUp() override {
    InterpreterWithTfProcessingTest::SetUp();
    const auto& [params, unused, lazy_loading_pipelines] = GetParam();
    create_options_->lazy_loading_pipelines = lazy_loading_pipelines;
    create_options_->model_name = params.model_name;
    create_options_->model_version = 0;
    create_options_->model_path =
        devtools_build::GetDataDependencyFilepath(params.model_path);
    create_options_->execute_worker_thread_pool =
        execute_worker_thread_pool_.get();
    create_options_->ifrt_client = client_;
    create_options_->computation_thread_pool = computation_thread_pool_.get();
    TF_ASSERT_OK_AND_ASSIGN(interpreter_,
                            Interpreter::Create(std::move(create_options_)));
  }
};

// Parameterized test to test both sync and async run.
class InterpreterLoadAndRunTest : public InterpreterWithTfProcessingTest,
                                  public testing::WithParamInterface<
                                      testing::tuple<absl::string_view, bool>> {
 protected:
  void SetUp() override {
    InterpreterWithTfProcessingTest::SetUp();
    const auto& [model_path, _] = GetParam();
    create_options_->model_name = "test_model";
    create_options_->model_version = 0;
    create_options_->model_path =
        devtools_build::GetDataDependencyFilepath(model_path);
    create_options_->execute_worker_thread_pool =
        execute_worker_thread_pool_.get();
    create_options_->ifrt_client = client_;
    create_options_->computation_thread_pool = computation_thread_pool_.get();
    TF_ASSERT_OK_AND_ASSIGN(interpreter_,
                            Interpreter::Create(std::move(create_options_)));
  }
};

class IdentityMultiIoModelTest
    : public InterpreterWithTfProcessingTest,
      public testing::WithParamInterface<
          testing::tuple<absl::string_view, /*enable_async_run=*/bool,
                         /*lazy_loading_pipelines=*/bool>> {
 protected:
  void SetUp() override {
    InterpreterWithTfProcessingTest::SetUp();
    const auto& [model_path, unused, lazy_loading_pipelines] = GetParam();
    create_options_->lazy_loading_pipelines = lazy_loading_pipelines;
    create_options_->model_name = "identity_multi_io_model";
    create_options_->model_version = 0;
    create_options_->model_path =
        devtools_build::GetDataDependencyFilepath(model_path);
    create_options_->execute_worker_thread_pool =
        execute_worker_thread_pool_.get();
    create_options_->ifrt_client = client_;
    create_options_->computation_thread_pool = computation_thread_pool_.get();
    TF_ASSERT_OK_AND_ASSIGN(interpreter_,
                            Interpreter::Create(std::move(create_options_)));
  }

  std::string pipeline_name_ = "identity";

  std::vector<Value> inputs_ = {
      {Value(AsTensor<float>({0.0, 1.0, 2.0, 3.0},
                             tensorflow::TensorShape({2, 2}))),
       Value(AsTensor<float>({4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0},
                             tensorflow::TensorShape({2, 4})))},
  };
  std::vector<testing::Matcher<const Value&>> expected_outputs_ = {
      // The additional parameter is added to the output.
      ValueOfType<tensorflow::Tensor>(TensorEq(
          AsTensor<float>({100.0, 100.0}, tensorflow::TensorShape({2})))),
      ValueOfType<tensorflow::Tensor>(TensorEq(AsTensor<float>(
          {0.0, 1.0, 2.0, 3.0}, tensorflow::TensorShape({2, 2})))),
      ValueOfType<tensorflow::Tensor>(
          TensorEq(AsTensor<float>({4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0},
                                   tensorflow::TensorShape({2, 4})))),
  };
};

class InterpreterWithTfProcessingMapIoTest
    : public InterpreterWithTfProcessingTest,
      public testing::WithParamInterface<InterpreterMapIoTestParam> {
 protected:
  void SetUp() override {
    InterpreterWithTfProcessingTest::SetUp();
    const auto& params = GetParam();
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

    create_options_->model_name = params.model_name;
    create_options_->model_version = 0;
    create_options_->model_path =
        devtools_build::GetDataDependencyFilepath(params.model_path);
    create_options_->execute_worker_thread_pool =
        execute_worker_thread_pool_.get();
    create_options_->ifrt_client = client_;
    create_options_->computation_thread_pool = computation_thread_pool_.get();
    TF_ASSERT_OK_AND_ASSIGN(interpreter_,
                            Interpreter::Create(std::move(create_options_)));
  }
};

TEST_P(InterpreterWithTfProcessingMatchOutputTest, LoadAndRunSucceeds) {
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

INSTANTIATE_TEST_SUITE_P(
    InterpreterMatchOutputTestInstances,
    InterpreterWithTfProcessingMatchOutputTest,
    testing::Combine(
        testing::ValuesIn<InterpreterMatchOutputTestParam>({
            // Tests a model with TF processing.
            // The model implements:
            //   - a preprocessing step: y = x + 1,
            //   - a simple linear transformation: y = x * w + b,
            //   - and a postprocessing step: y = x * 2 + 1.
            // weights are
            // Array([[1., 1.],
            //        [1., 1.]], dtype=float32)
            // bias are
            // Array([0., 0.], dtype=float32)
            {
                .model_path = kSimpleModelWithTfProcessingPath,
                .pipeline_name = "serving_default",
                .inputs = {Value(
                    AsTensor<float>({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0},
                                    tensorflow::TensorShape({4, 2})))},
                .expected_outputs = {ValueOfType<tensorflow::Tensor>(TensorEq(
                    AsTensor<float>({7.0, 7.0, 15.0, 15.0, 23.0, 23.0, 31.0,
                                     31.0},
                                    tensorflow::TensorShape({4, 2}))))},
            },
            // Same as above, but use a different batch size to test
            // polymorphic batch size.
            {
                .model_path = kSimpleModelWithTfProcessingPath,
                .pipeline_name = "serving_default",
                .inputs = {Value(AsTensor<float>(
                    {8.0, 9.0, 10.0, 11.0}, tensorflow::TensorShape({2, 2})))},
                .expected_outputs = {ValueOfType<tensorflow::Tensor>(TensorEq(
                    AsTensor<float>({39.0, 39.0, 47.0, 47.0},
                                    tensorflow::TensorShape({2, 2}))))},
            },
            // Same as above, but use TF resources to implement the
            // preprocessing and postprocessing steps.
            {
                .model_path = kSimpleModelWithTfResourcesPath,
                .pipeline_name = "linear",
                .inputs = {Value(
                    AsTensor<float>({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0},
                                    tensorflow::TensorShape({4, 2})))},
                .expected_outputs = {ValueOfType<tensorflow::Tensor>(TensorEq(
                    AsTensor<float>({7.0, 7.0, 15.0, 15.0, 23.0, 23.0, 31.0,
                                     31.0},
                                    tensorflow::TensorShape({4, 2}))))},
            },
            // Tests the unsharded version of `SimpleModelWithTfProcessing`.
            {
                .model_path = kSimpleModelWithTfProcessingUnshardedPath,
                .pipeline_name = "serving_default",
                .inputs = {Value(
                    AsTensor<float>({12.0, 13.0, 14.0, 15.0, 16.0, 17.0},
                                    tensorflow::TensorShape({3, 2})))},
                .expected_outputs = {ValueOfType<tensorflow::Tensor>(TensorEq(
                    AsTensor<float>({55.0, 55.0, 63.0, 63.0, 71.0, 71.0},
                                    tensorflow::TensorShape({3, 2}))))},
            },
            // Tests a model with TF pre-processing but no post-processing.
            // The model implements:
            //   - a preprocessing step: x' = x + 1,
            //   - a simple linear transformation: y = x' * w + b,
            // weights are
            // Array([[1., 1.],
            //        [1., 1.]], dtype=float32)
            // bias are
            // Array([0., 0.], dtype=float32)
            {
                .model_path = kSimpleModelWithTfProcessingPath,
                .pipeline_name = "preprocessor_and_model_fn",
                .inputs = {Value(
                    AsTensor<float>({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0},
                                    tensorflow::TensorShape({4, 2})))},
                .expected_outputs = {ValueOfType<HostBuffer>(HostBufferEq(
                    AsHostBuffer<float>({{3.0, 3.0, 7.0, 7.0, 11.0, 11.0, 15.0,
                                          15.0}},
                                        xla::ifrt::Shape({4, 2}))))},
            },
            // Tests a model with TF post-processing but no pre-processing.
            // The model implements:
            //   - a simple linear transformation: y = x * w + b,
            //   - and a postprocessing step: y' = y * 2 + 1.
            // weights are
            // Array([[1., 1.],
            //        [1., 1.]], dtype=float32)
            // bias are
            // Array([0., 0.], dtype=float32)
            {
                .model_path = kSimpleModelWithTfProcessingPath,
                .pipeline_name = "model_fn_and_postprocessor",
                .inputs = {Value(AsHostBuffer<float>(
                    {{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}},
                    xla::ifrt::Shape({4, 2})))},
                .expected_outputs = {ValueOfType<tensorflow::Tensor>(TensorEq(
                    AsTensor<float>({3.0, 3.0, 11.0, 11.0, 19.0, 19.0, 27.0,
                                     27.0},
                                    tensorflow::TensorShape({4, 2}))))},
            },
            // Tests a model with TF preprocessor only.
            // The model implements:
            //   - a preprocessing step: y = x + 1,
            {
                .model_path = kSimpleModelWithTfPreprocessorOnlyPath,
                .pipeline_name = "preprocessor_only",
                .inputs = {Value(
                    AsTensor<float>({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0},
                                    tensorflow::TensorShape({4, 2})))},
                .expected_outputs = {ValueOfType<HostBuffer>(HostBufferEq(
                    AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                         8.0},
                                        xla::ifrt::Shape({4, 2}))))},
            },
            // Tests a model that decodes a string to a float, multiplies it by
            // a factor (2.0), and then encodes it back to a string.
            {
                .model_path = kStringToNumberRoundTripModelPath,
                .pipeline_name = "multiply_by_factor",
                .inputs = {Value(
                    AsTensor<tsl::tstring>({"1.0", "2.0", "3.0", "4.0"},
                                           tensorflow::TensorShape({4})))},
                .expected_outputs = {ValueOfType<tensorflow::Tensor>(TensorEq(
                    AsTensor<tsl::tstring>({"2", "4", "6", "8"},
                                           tensorflow::TensorShape({4}))))},
            },
        }),
        /*enable_async_run=*/testing::Bool(),
        /*lazy_loading_pipelines=*/testing::Bool()));

// Tests a model with multiple inputs and outputs in each step.
// All computations are identity functions. The JAX function will prepend an
// additional parameter to the output.
TEST_P(IdentityMultiIoModelTest, LoadAndRunSucceeds) {
  // Use a vector-based input and output.
  EXPECT_THAT(interpreter_->Run(pipeline_name_, absl::MakeSpan(inputs_)),
              IsOkAndHolds(ElementsAreArray(expected_outputs_)));
}

TEST_P(IdentityMultiIoModelTest, LoadAndRunWithMapInputSucceeds) {
  // Use a map-based input and vector-based output.
  std::vector<std::string> input_names;
  input_names = {"a", "b"};

  absl::flat_hash_map<std::string, Value> keyword_inputs;
  for (int i = 0; i < inputs_.size(); ++i) {
    keyword_inputs[input_names[i]] = inputs_[i];
  }

  const auto [model_path, enable_async_run, unused] = GetParam();

  absl::StatusOr<std::vector<Value>> outputs;
  if (enable_async_run) {
    outputs = interpreter_
                  ->RunAsync(pipeline_name_,
                             /*positional_inputs=*/{}, keyword_inputs)
                  .Await();
  } else {
    outputs = interpreter_->Run(pipeline_name_,
                                /*positional_inputs=*/{}, keyword_inputs);
  }
  EXPECT_THAT(outputs, IsOkAndHolds(ElementsAreArray(expected_outputs_)));
}

TEST_P(IdentityMultiIoModelTest, LoadAndRunWithMapInputAndOutputSucceeds) {
  // Use a map-based input and output.
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  input_names = {"a", "b"};

  const auto [model_path, enable_async_run, unused] = GetParam();
  if (model_path == kIdentityMultiIoModelUseTuplePath) {
    // Positional outputs have their names sourced from the TF saved model.
    output_names = {"output_0", "output_1", "output_2"};
  } else {
    output_names = {"out0", "out1", "out2"};
  }

  absl::flat_hash_map<std::string, Value> keyword_inputs;
  for (int i = 0; i < inputs_.size(); ++i) {
    keyword_inputs[input_names[i]] = inputs_[i];
  }

  std::vector<testing::Matcher<std::pair<const std::string, const Value&>>>
      expected_map_outputs;
  for (int i = 0; i < expected_outputs_.size(); ++i) {
    expected_map_outputs.push_back(Pair(output_names[i], expected_outputs_[i]));
  }

  absl::StatusOr<absl::flat_hash_map<std::string, Value>> map_outputs;
  if (enable_async_run) {
    map_outputs =
        interpreter_
            ->RunAsyncWithMapReturn(pipeline_name_,
                                    /*positional_inputs=*/{}, keyword_inputs)
            .Await();
  } else {
    map_outputs = interpreter_->RunWithMapReturn(pipeline_name_,
                                                 /*positional_inputs=*/{},
                                                 keyword_inputs);
  }
  EXPECT_THAT(map_outputs,
              IsOkAndHolds(UnorderedElementsAreArray(expected_map_outputs)));
}

// Test different ways to write the same model.
INSTANTIATE_TEST_SUITE_P(
    IdentityMultiIoModelTestInstances, IdentityMultiIoModelTest,
    testing::Combine(/*model_path=*/
                     testing::Values(
                         kIdentityMultiIoModelUseTuplePath,
                         // The following models pass when inputs or outputs are
                         // vector-based because when inputs and outputs are
                         // flattened, they are ordered by keys (using string
                         // comparison) in the dict.
                         kIdentityMultiIoModelUseDictPath,
                         kIdentityMultiIoModelUseDictUnshardedPath,
                         kIdentityMultiIoModelKwargsOnlyPath,
                         kIdentityMultiIoModelKwargsOnlyUnshardedPath),
                     /*enable_async_run=*/testing::Bool(),
                     /*lazy_loading_pipelines=*/testing::Bool()));

TEST_P(InterpreterLoadAndRunTest, MnistModel) {
  // This model implements the following:
  //   - a preprocessing step: read RBG images, convert them to grayscale, and
  //   resize them to 28x28,
  //   - a prediction step with the mnist model,
  //   - and a postprocessing step: select the digit with the highest
  //   probability for each image
  const auto [_, enable_async_run] = GetParam();
  absl::string_view pipeline_name = "mnist_forward_fn";

  int batch_size = 4;
  // Simulate a batch of 4 images of size 500x500x3.
  tensorflow::TensorShape x_shape =
      tensorflow::TensorShape({batch_size, 500, 500, 3});
  const int64_t kInputSize = x_shape.num_elements();
  std::vector<uint8_t> x_data;
  x_data.reserve(kInputSize);
  for (int i = 0; i < kInputSize; ++i) {
    x_data.push_back(i % 256);
  }

  auto x = AsTensor<uint8_t>(x_data, x_shape);
  std::vector<Value> inputs = {Value(x)};
  std::vector<Value> outputs;

  if (!enable_async_run) {
    TF_ASSERT_OK_AND_ASSIGN(
        outputs, interpreter_->Run(pipeline_name, absl::MakeSpan(inputs)));
  } else {
    tsl::Future<std::vector<Value>> outputs_future =
        interpreter_->RunAsync(pipeline_name, absl::MakeSpan(inputs));
    TF_ASSERT_OK_AND_ASSIGN(outputs, outputs_future.Await());
  }

  ASSERT_EQ(outputs.size(), 1);
  ASSERT_TRUE(outputs[0].IsType<tensorflow::Tensor>());
  auto output_data = outputs[0].Get<tensorflow::Tensor>();
  EXPECT_EQ(output_data.dtype(), tensorflow::DT_INT64);
  EXPECT_EQ(output_data.shape(), tensorflow::TensorShape({batch_size}));
  EXPECT_THAT(absl::MakeSpan(output_data.flat<int64_t>().data(),
                             output_data.NumElements()),
              Each(Eq(1)));
}

INSTANTIATE_TEST_SUITE_P(
    InterpreterLoadAndRunTestInstances, InterpreterLoadAndRunTest,
    testing::Combine(/*model_path=*/testing::Values(kMnistModelPath),
                     /*enable_async_run=*/testing::Bool()));

TEST_P(InterpreterWithTfProcessingMapIoTest, LoadAndRunSucceedsWithMapReturn) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto outputs, interpreter_->RunWithMapReturn(GetParam().pipeline_name,
                                                   /*positional_inputs=*/{},
                                                   GetParam().inputs));
  EXPECT_THAT(outputs, UnorderedElementsAreArray(GetParam().expected_outputs));
}

INSTANTIATE_TEST_SUITE_P(
    InterpreterWithTfProcessingMapIoTestInstances,
    InterpreterWithTfProcessingMapIoTest,
    testing::ValuesIn<InterpreterMapIoTestParam>({
        {
            .model_path = kSimpleModelWithTfResourcesPath,
            .pipeline_name = "linear",
            .inputs = {{"x", Value(AsTensor<float>(
                                 {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0},
                                 tensorflow::TensorShape({4, 2})))}},
            .expected_outputs = {Pair(
                "y", ValueOfType<tensorflow::Tensor>(TensorEq(AsTensor<float>(
                         {7.0, 7.0, 15.0, 15.0, 23.0, 23.0, 31.0, 31.0},
                         tensorflow::TensorShape({4, 2})))))},
        },
        {
            .model_path = kSimpleModelWithTfProcessingPath,
            .pipeline_name = "serving_default",
            .inputs = {{"new_a", Value(AsTensor<float>(
                                     {8.0, 1.0, 2.0, 9.0},
                                     tensorflow::TensorShape({2, 2})))}},
            .expected_outputs = {Pair(
                "out", ValueOfType<tensorflow::Tensor>(TensorEq(
                           AsTensor<float>({23.0, 23.0, 27.0, 27.0},
                                           tensorflow::TensorShape({2, 2})))))},
        },
    }));

TEST_F(InterpreterWithTfProcessingTest, AsyncFailureCanReturn) {
  // The model expects a [4, 2] input.
  const std::string kModelPath = devtools_build::GetDataDependencyFilepath(
      kSimpleModelWithTfProcessingPath);
  auto create_options =
      std::make_unique<TfCompatibleInterpreterCreateOptions>();
  create_options->model_name = "test_model";
  create_options->model_version = 0;
  create_options->model_path = kModelPath;
  create_options->execute_worker_thread_pool =
      execute_worker_thread_pool_.get();
  create_options->ifrt_client = client_;
  create_options->computation_thread_pool = computation_thread_pool_.get();
  TF_ASSERT_OK_AND_ASSIGN(interpreter_,
                          Interpreter::Create(std::move(create_options)));
  // Shape should be [?, 2], but intentionally set it to a different shape to
  // trigger failure.
  auto x = AsTensor<float>({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0},
                           tensorflow::TensorShape({2, 4}));
  std::vector<Value> inputs({Value(x)});
  tsl::Future<std::vector<Value>> output_futures = interpreter_->RunAsync(
      /*pipeline_name=*/"serving_default", absl::MakeSpan(inputs));
  EXPECT_THAT(output_futures.Await(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("does not match the expected shape")));
}

TEST_F(InterpreterWithTfProcessingTest, ConcurrentLoadAndRunSucceeds) {
  // Load two models concurrently to make sure the loading does not interfere
  // with each other.
  std::vector<std::string> model_paths = {
      devtools_build::GetDataDependencyFilepath(
          kSimpleModelWithTfProcessingPath),
      devtools_build::GetDataDependencyFilepath(kMnistModelPath)};
  int num_models = model_paths.size();

  std::vector<std::unique_ptr<Interpreter>> interpreters(num_models);

  std::vector<std::string> pipeline_names = {"serving_default",
                                             "mnist_forward_fn"};
  std::vector<std::vector<Value>> inputs(num_models);
  inputs[0] = {Value(AsTensor<float>({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0},
                                     tensorflow::TensorShape({4, 2})))};
  tensorflow::TensorShape x_shape = tensorflow::TensorShape({4, 3, 3, 3});
  const int64_t kInputSize = x_shape.num_elements();
  std::vector<uint8_t> x_data;
  x_data.reserve(kInputSize);
  for (int i = 0; i < kInputSize; ++i) {
    x_data.push_back(i % 256);
  }
  inputs[1] = {Value(AsTensor<uint8_t>(x_data, x_shape))};

  const std::string platform_name_override =
      absl::GetFlag(FLAGS_jsv_testonly_platform_name_override);
  std::vector<std::vector<Value>> outputs(num_models);
  auto thread_fn = [&](int i) -> absl::Status {
    auto create_options =
        std::make_unique<TfCompatibleInterpreterCreateOptions>();
    if (!platform_name_override.empty()) {
      create_options->platform_name = platform_name_override;
    }
    create_options->model_name = absl::StrCat("test_model_", i);
    create_options->model_version = 0;
    create_options->model_path = model_paths[i];
    create_options->execute_worker_thread_pool =
        execute_worker_thread_pool_.get();
    create_options->ifrt_client = client_;
    create_options->computation_thread_pool = computation_thread_pool_.get();
    TF_ASSIGN_OR_RETURN(interpreters[i],
                        Interpreter::Create(std::move(create_options)));
    TF_ASSIGN_OR_RETURN(
        outputs[i],
        interpreters[i]->Run(pipeline_names[i], absl::MakeSpan(inputs[i])));
    return absl::OkStatus();
  };

  std::vector<absl::Status> statuses(num_models, absl::UnknownError("Not run"));
  {
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_threads",
                                        num_models);
    for (int i = 0; i < num_models; ++i) {
      thread_pool.Schedule([&, i]() { statuses[i] = thread_fn(i); });
    }
  }

  for (int i = 0; i < num_models; ++i) {
    TF_ASSERT_OK(statuses[i]);
  }

  EXPECT_THAT(
      outputs[0],
      ElementsAreArray({ValueOfType<tensorflow::Tensor>(TensorEq(
          AsTensor<float>({7.0, 7.0, 15.0, 15.0, 23.0, 23.0, 31.0, 31.0},
                          tensorflow::TensorShape({4, 2}))))}));

  // With untrained weights, the output is somewhat random. Here we check
  // that the output is a valid int64 tensor of shape [4] and the results are
  // deterministic.
  EXPECT_THAT(
      outputs[1],
      ElementsAreArray({ValueOfType<tensorflow::Tensor>(TensorEq(
          AsTensor<int64_t>({6, 1, 1, 1}, tensorflow::TensorShape({4}))))}));
}

// Parameterized test to load and run the model and check the outputs.
class InterpreterWithTfProcessingMultiPipelineRunTest
    : public InterpreterWithTfProcessingTest,
      public testing::WithParamInterface<testing::tuple<
          /*enable_async_run=*/bool, /*lazy_loading_pipelines=*/bool>> {
 protected:
  void SetUp() override {
    InterpreterWithTfProcessingTest::SetUp();
    const auto [unused, lazy_loading_pipelines] = GetParam();
    create_options_->lazy_loading_pipelines = lazy_loading_pipelines;
    create_options_->model_name = "test_model_with_multiple_pipelines";
    create_options_->model_version = 0;
    create_options_->model_path = devtools_build::GetDataDependencyFilepath(
        kSimpleModelWithTfProcessingPath);
    create_options_->execute_worker_thread_pool =
        execute_worker_thread_pool_.get();
    create_options_->ifrt_client = client_;
    create_options_->computation_thread_pool = computation_thread_pool_.get();
    TF_ASSERT_OK_AND_ASSIGN(interpreter_,
                            Interpreter::Create(std::move(create_options_)));
  }

  // Use a thread pool to simulate concurrent runs.
  tsl::thread::ThreadPool thread_pool_ =
      tsl::thread::ThreadPool(tsl::Env::Default(), "test_threads", 32);
};

TEST_P(InterpreterWithTfProcessingMultiPipelineRunTest, RunSucceeds) {
  const auto [enable_async_run, unused] = GetParam();
  const std::vector<
      std::tuple<std::string, Value, testing::Matcher<const Value&>>>
      inputs_and_expected_outputs{
          {"serving_default",
           Value(AsTensor<float>({0.0, 1.0, 2.0, 3.0},
                                 tensorflow::TensorShape({2, 2}))),
           ValueOfType<tensorflow::Tensor>(TensorEq(AsTensor<float>(
               {7.0, 7.0, 15.0, 15.0}, tensorflow::TensorShape({2, 2}))))},
          {"preprocessor_and_model_fn",
           Value(AsTensor<float>({4.0, 5.0, 6.0, 7.0},
                                 tensorflow::TensorShape({2, 2}))),
           ValueOfType<HostBuffer>(HostBufferEq(AsHostBuffer<float>(
               {{11.0, 11.0, 15.0, 15.0}}, xla::ifrt::Shape({2, 2}))))},
          {"model_fn_and_postprocessor",
           Value(AsHostBuffer<float>({{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}},
                                     xla::ifrt::Shape({4, 2}))),
           ValueOfType<tensorflow::Tensor>(TensorEq(
               AsTensor<float>({3.0, 3.0, 11.0, 11.0, 19.0, 19.0, 27.0, 27.0},
                               tensorflow::TensorShape({4, 2}))))},
      };

  for (int i = 0; i < 100; ++i) {
    for (const auto& [pipeline_name, input, expected_output] :
         inputs_and_expected_outputs) {
      thread_pool_.Schedule(
          [&, enable_async_run, pipeline_name, input, expected_output] {
            std::vector<Value> inputs = {input};
            absl::StatusOr<std::vector<Value>> outputs =
                enable_async_run
                    ? interpreter_->RunAsync(pipeline_name, inputs).Await()
                    : interpreter_->Run(pipeline_name, inputs);
            EXPECT_THAT(outputs, IsOkAndHolds(ElementsAre(expected_output)));
          });
    }
  }
}

TEST_P(InterpreterWithTfProcessingMultiPipelineRunTest,
       RunWithMapPartialFailure) {
  const auto [enable_async_run, unused] = GetParam();
  std::vector<std::tuple<std::string, absl::flat_hash_map<std::string, Value>,
                         absl::StatusOr<std::vector<
                             testing::Matcher<std::pair<std::string, Value>>>>>>
      inputs_and_expected_outputs{
          {"serving_default",
           {{"new_a", Value(AsTensor<float>({0.0, 1.0, 2.0, 3.0},
                                            tensorflow::TensorShape({2, 2})))}},
           std::vector<testing::Matcher<std::pair<std::string, Value>>>{Pair(
               "out", ValueOfType<tensorflow::Tensor>(TensorEq(
                          AsTensor<float>({7.0, 7.0, 15.0, 15.0},
                                          tensorflow::TensorShape({2, 2})))))}},
          {"preprocessor_and_model_fn",
           {{"new_a", Value(AsTensor<float>({4.0, 5.0, 6.0, 7.0},
                                            tensorflow::TensorShape({2, 2})))}},
           // Named output not allowed.
           absl::FailedPreconditionError("")},
          {"model_fn_and_postprocessor",
           {{"x", Value(AsHostBuffer<float>(
                      {{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}},
                      xla::ifrt::Shape({4, 2})))}},
           // Named input not allowed.
           absl::FailedPreconditionError("")},
      };
  for (int i = 0; i < 100; ++i) {
    for (const auto& [pipeline_name, inputs, expected_output] :
         inputs_and_expected_outputs) {
      thread_pool_.Schedule([&, enable_async_run, pipeline_name, inputs,
                             expected_output] {
        absl::StatusOr<absl::flat_hash_map<std::string, Value>> outputs =
            enable_async_run
                ? interpreter_
                      ->RunAsyncWithMapReturn(pipeline_name,
                                              /*positional_inputs=*/{}, inputs)
                      .Await()
                : interpreter_->RunWithMapReturn(pipeline_name,
                                                 /*positional_inputs=*/{},
                                                 inputs);
        ASSERT_EQ(outputs.status().code(), expected_output.status().code());
        if (expected_output.ok()) {
          EXPECT_THAT(*outputs, UnorderedElementsAreArray(*expected_output));
        }
      });
    }
  }
}

TEST_P(InterpreterWithTfProcessingMultiPipelineRunTest,
       RunNewPipelineAfterFullyInitialized) {
  const auto [enable_async_run, lazy_loading_pipelines] = GetParam();
  std::vector<Value> inputs = {Value(
      AsTensor<float>({0.0, 1.0, 2.0, 3.0}, tensorflow::TensorShape({2, 2})))};
  ASSERT_OK(interpreter_->Run("serving_default", inputs));
  ASSERT_OK(interpreter_->MarkFullyInitialized());
  // Run the same pipeline again to make sure it succeeds.
  ASSERT_OK(interpreter_->Run("serving_default", inputs));
  // Run a new pipeline after fully initialized. It should fail if lazy loading
  // is enabled.
  absl::StatusOr<std::vector<Value>> output =
      enable_async_run
          ? interpreter_->RunAsync("preprocessor_and_model_fn", inputs).Await()
          : interpreter_->Run("preprocessor_and_model_fn", inputs);
  if (!lazy_loading_pipelines) {
    EXPECT_THAT(
        output,
        StatusIs(absl::StatusCode::kFailedPrecondition,
                 HasSubstr("new input shapes requiring a new executable")));
  } else {
    EXPECT_THAT(output, StatusIs(absl::StatusCode::kFailedPrecondition,
                                 HasSubstr("not loaded during warmup")));
  }
  // Run a non-existent pipeline after fully initialized. It should fail.
  EXPECT_THAT(interpreter_->Run("non_existent_pipeline", inputs),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

INSTANTIATE_TEST_SUITE_P(
    InterpreterWithTfProcessingMultiPipelineCurrentRunTestInstances,
    InterpreterWithTfProcessingMultiPipelineRunTest,
    testing::Combine(/*enable_async_run=*/testing::Bool(),
                     /*lazy_loading_pipelines=*/testing::Bool()));

class InterpreterWithTfProcessingGraphOrchestrationTest
    : public InterpreterWithTfProcessingTest {
 protected:
  void SetUp() override {
    InterpreterWithTfProcessingTest::SetUp();
    create_options_->model_name = "graph_orchestration";
    create_options_->model_version = 0;
    create_options_->model_path =
        devtools_build::GetDataDependencyFilepath(kGraphOrchestrationModelPath);
    create_options_->execute_worker_thread_pool =
        execute_worker_thread_pool_.get();
    create_options_->ifrt_client = client_;
    create_options_->computation_thread_pool = computation_thread_pool_.get();
    TF_ASSERT_OK_AND_ASSIGN(interpreter_,
                            Interpreter::Create(std::move(create_options_)));
  }
};

TEST_F(InterpreterWithTfProcessingGraphOrchestrationTest,
       RunReturnsUnimplemented) {
  EXPECT_THAT(interpreter_->Run(
                  /*pipeline_name=*/"linear_model_with_graph_orchestration",
                  /*positional_inputs=*/{},
                  /*keyword_inputs=*/{}),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("ExecuteLinearGraphWithKeywordArgs")));
}

}  // namespace
}  // namespace jsv
