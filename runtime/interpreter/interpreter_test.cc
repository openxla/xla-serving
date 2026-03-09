#include "third_party/neptune/runtime/interpreter/interpreter.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "devtools/build/runtime/get_runfiles_dir.h"
#include "third_party/neptune/runtime/interpreter/interpreter_create_options.h"
#include "third_party/neptune/runtime/interpreter/interpreter_test_base.h"
#include "third_party/neptune/runtime/interpreter/optional_function_handler_factory.h"
#include "third_party/neptune/runtime/interpreter/run_options.h"
#include "third_party/neptune/runtime/interpreter/test_util.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "testing/base/public/mock-log.h"
#include "third_party/absl/base/log_severity.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/flags/flag.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/time/clock.h"
#include "third_party/absl/time/time.h"
#include "third_party/absl/types/span.h"
#include "third_party/neptune/runtime/core/constants.h"
#include "third_party/neptune/runtime/core/flat_signature.h"
#include "third_party/neptune/runtime/core/host_buffer.h"
#include "third_party/neptune/runtime/core/host_buffer_matcher.h"
#include "third_party/neptune/runtime/core/host_buffer_test_utils.h"
#include "third_party/neptune/runtime/core/stablehlo_utils.h"
#include "third_party/neptune/runtime/core/value.h"
#include "third_party/neptune/runtime/core/value_matcher.h"
#include "third_party/py/orbax/experimental/model/core/protos/manifest.proto.h"
#include "third_party/py/orbax/export/protos/oex_orchestration.proto.h"
#include "third_party/tensorflow/compiler/xla/hlo/ir/hlo_sharding.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/dtype.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/shape.h"
#include "third_party/tensorflow/compiler/xla/tsl/concurrency/future.h"
#include "third_party/tensorflow/compiler/xla/tsl/lib/core/status_test_util.h"
#include "third_party/tensorflow/compiler/xla/tsl/lib/monitoring/cell_reader.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"
#include "third_party/tensorflow/compiler/xla/xla_data.proto.h"

namespace jsv {

class InterpreterPeer {
 public:
  static absl::StatusOr<KernelGraph> LoadGraph(
      Interpreter* interpreter,
      const orbax_export_orchestration::Pipeline& pipeline) {
    return interpreter->LoadGraph(pipeline);
  }
};

namespace {
using ::jsv::test_util::MetaArgSpecsAre;
using ::testing::_;
using ::testing::AnyNumber;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::kDoNotCaptureLogsYet;
using ::testing::Pair;
using ::testing::ScopedMockLog;
using ::testing::UnorderedElementsAre;
using ::testing::UnorderedElementsAreArray;
using ::testing::status::StatusIs;
using ::tsl::monitoring::testing::CellReader;

constexpr absl::string_view kXTimesYModelPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/x_times_y_gen";
constexpr absl::string_view kLinearModelPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/linear_model_gen";
constexpr absl::string_view kMakeNumberFromDigitsDictModelPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/make_number_from_digits_dict_gen";
constexpr absl::string_view kDynamicShapePath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/dynamic_shape_gen";
constexpr absl::string_view kDynamicShapeWithBatchingPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/dynamic_shape_with_batching_gen";
constexpr absl::string_view kLinearModelPolymorphicPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/linear_model_polymorphic_gen";
constexpr absl::string_view kMultipleApplyFnSharingCkptPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/multiple_apply_fn_sharing_ckpt_gen";
constexpr absl::string_view kSimpleModelWithTfProcessingPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/simple_model_with_tf_processing_gen";

///////////////////////////
///// InterpreterTest /////
///////////////////////////

class InterpreterTest : public InterpreterTestBase {
 protected:
  void SetUp() override {
    InterpreterTestBase::SetUp();
    create_options_ = std::make_unique<InterpreterCreateOptions>();
    if (!absl::GetFlag(FLAGS_jsv_testonly_platform_name_override).empty()) {
      orbax_model_manifest::Platform platform;
      ASSERT_TRUE(orbax_model_manifest::Platform_Parse(
          absl::GetFlag(FLAGS_jsv_testonly_platform_name_override), &platform));
      create_options_->platform_name =
          orbax_model_manifest::Platform_Name(platform);
    }
    optional_function_handler_router_ =
        std::make_unique<OptionalFunctionHandlerRouter>();
  }

 protected:
  std::unique_ptr<InterpreterCreateOptions> create_options_;
  std::unique_ptr<Interpreter> interpreter_;
  std::unique_ptr<OptionalFunctionHandlerRouter>
      optional_function_handler_router_;
};

TEST_F(InterpreterTest, MapInputNotSupported) {
  create_options_->model_name = "x_times_y";
  create_options_->model_version = 0;
  create_options_->model_path =
      devtools_build::GetDataDependencyFilepath(kXTimesYModelPath);
  create_options_->execute_worker_thread_pool =
      execute_worker_thread_pool_.get();
  create_options_->ifrt_client = client_;
  create_options_->computation_thread_pool = computation_thread_pool_.get();
  TF_ASSERT_OK_AND_ASSIGN(auto interpreter,
                          Interpreter::Create(std::move(create_options_)));
  const absl::flat_hash_map<std::string, Value> inputs = {
      {"x", Value(AsHostBuffer<float>({{0.0, 1.0, 2.0, 3.0}},
                                      xla::ifrt::Shape({2, 2})))},
      {"y", Value(AsHostBuffer<float>({{0.0, 1.0, 2.0, 3.0}},
                                      xla::ifrt::Shape({2, 2})))},
  };
  // The input signature is not map-based.
  // Run synchronously.
  EXPECT_THAT(interpreter->Run(
                  /*pipeline_name=*/"x_times_y",
                  /*positional_inputs=*/{}, /*keyword_inputs=*/inputs),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("cannot run this pipeline with a map")));
  // Run asynchronously.
  EXPECT_THAT(interpreter
                  ->RunAsync(
                      /*pipeline_name=*/"x_times_y",
                      /*positional_inputs=*/{}, /*keyword_inputs=*/inputs)
                  .Await(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("cannot run this pipeline with a map")));
}

TEST_F(InterpreterTest, MapOutputNotSupported) {
  create_options_->model_name = "x_times_y";
  create_options_->model_version = 0;
  create_options_->model_path =
      devtools_build::GetDataDependencyFilepath(kXTimesYModelPath);
  create_options_->execute_worker_thread_pool =
      execute_worker_thread_pool_.get();
  create_options_->ifrt_client = client_;
  create_options_->computation_thread_pool = computation_thread_pool_.get();
  TF_ASSERT_OK_AND_ASSIGN(auto interpreter,
                          Interpreter::Create(std::move(create_options_)));
  const std::vector<Value> inputs = {
      Value(AsHostBuffer<float>({{0.0, 1.0, 2.0, 3.0}},
                                xla::ifrt::Shape({2, 2}))),
      Value(AsHostBuffer<float>({{0.0, 1.0, 2.0, 3.0}},
                                xla::ifrt::Shape({2, 2})))};
  // The output signatures is not map-based.
  // Run synchronously.
  EXPECT_THAT(interpreter->RunWithMapReturn(
                  /*pipeline_name=*/"x_times_y",
                  /*positional_inputs=*/inputs, /*keyword_inputs=*/{}),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("map return is not supported")));
  // Run asynchronously.
  EXPECT_THAT(interpreter
                  ->RunAsyncWithMapReturn(
                      /*pipeline_name=*/"x_times_y",
                      /*positional_inputs=*/inputs, /*keyword_inputs=*/{})
                  .Await(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("map return is not supported")));
}

TEST_F(InterpreterTest, MissingInput) {
  create_options_->model_name = "make_number_from_digits_dict";
  create_options_->model_version = 0;
  create_options_->model_path = devtools_build::GetDataDependencyFilepath(
      kMakeNumberFromDigitsDictModelPath);
  create_options_->execute_worker_thread_pool =
      execute_worker_thread_pool_.get();
  create_options_->ifrt_client = client_;
  create_options_->computation_thread_pool = computation_thread_pool_.get();
  TF_ASSERT_OK_AND_ASSIGN(auto interpreter,
                          Interpreter::Create(std::move(create_options_)));
  EXPECT_THAT(
      interpreter->RunWithMapReturn(
          /*pipeline_name=*/"make_number_from_digits",
          /*positional_inputs=*/{},
          {{"fourth_digit",
            Value(AsHostBuffer<int32_t>({4, 5, 6, 7, 8, 9, 3, 2},
                                        xla::ifrt::Shape({4, 2})))}}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("is not set and does not have a default value")));
}

TEST_F(InterpreterTest, WrongInputName) {
  create_options_->model_name = "make_number_from_digits_dict";
  create_options_->model_version = 0;
  create_options_->model_path = devtools_build::GetDataDependencyFilepath(
      kMakeNumberFromDigitsDictModelPath);
  create_options_->execute_worker_thread_pool =
      execute_worker_thread_pool_.get();
  create_options_->ifrt_client = client_;
  create_options_->computation_thread_pool = computation_thread_pool_.get();
  TF_ASSERT_OK_AND_ASSIGN(auto interpreter,
                          Interpreter::Create(std::move(create_options_)));
  EXPECT_THAT(interpreter->RunWithMapReturn(
                  /*pipeline_name=*/"make_number_from_digits",
                  /*positional_inputs=*/{},
                  {{"wrong_name",
                    Value(AsHostBuffer<int32_t>({4, 5, 6, 7, 8, 9, 3, 2},
                                                xla::ifrt::Shape({4, 2})))},
                   {"fourth_digit",
                    Value(AsHostBuffer<int32_t>({1, 2, 3, 4, 5, 6, 7, 8},
                                                xla::ifrt::Shape({4, 2})))},
                   {"fifth_digit",
                    Value(AsHostBuffer<int32_t>({9, 8, 7, 6, 5, 4, 3, 2},
                                                xla::ifrt::Shape({4, 2})))}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("name \"wrong_name\" is not found")));
}

TEST_F(InterpreterTest, AsyncWithMapReturnFailureCanReturn) {
  create_options_->model_name = "test_model";
  create_options_->model_version = 0;
  create_options_->model_path =
      devtools_build::GetDataDependencyFilepath(kLinearModelPolymorphicPath);
  create_options_->execute_worker_thread_pool =
      execute_worker_thread_pool_.get();
  create_options_->ifrt_client = client_;
  create_options_->computation_thread_pool = computation_thread_pool_.get();
  TF_ASSERT_OK_AND_ASSIGN(interpreter_,
                          Interpreter::Create(std::move(create_options_)));
  // Value is an int, but the model expects a HostBuffer.
  int x = 10;
  absl::flat_hash_map<std::string, Value> inputs({{"x", Value(x)}});
  tsl::Future<absl::flat_hash_map<std::string, Value>> output_futures =
      interpreter_->RunAsyncWithMapReturn(/*pipeline_name=*/"linear",
                                          /*positional_inputs=*/{}, inputs);
  EXPECT_THAT(output_futures.Await(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Value is not of type")));
}

TEST_F(InterpreterTest, GetPipelineNames) {
  create_options_->model_name = "linear_model";
  create_options_->model_version = 0;
  create_options_->model_path =
      devtools_build::GetDataDependencyFilepath(kLinearModelPath);
  create_options_->execute_worker_thread_pool =
      execute_worker_thread_pool_.get();
  create_options_->ifrt_client = client_;
  create_options_->computation_thread_pool = computation_thread_pool_.get();
  TF_ASSERT_OK_AND_ASSIGN(auto interpreter,
                          Interpreter::Create(std::move(create_options_)));
  EXPECT_THAT(interpreter->GetPipelineNames(),
              UnorderedElementsAre("linear", "serving_default"));
}

TEST_F(InterpreterTest, GetPipelineNamesMultiple) {
  create_options_->model_name = "multiple_apply_fn_sharing_ckpt";
  create_options_->model_version = 0;
  create_options_->model_path = devtools_build::GetDataDependencyFilepath(
      kMultipleApplyFnSharingCkptPath);
  create_options_->execute_worker_thread_pool =
      execute_worker_thread_pool_.get();
  create_options_->ifrt_client = client_;
  create_options_->computation_thread_pool = computation_thread_pool_.get();
  TF_ASSERT_OK_AND_ASSIGN(auto interpreter,
                          Interpreter::Create(std::move(create_options_)));
  EXPECT_THAT(
      interpreter->GetPipelineNames(),
      UnorderedElementsAre("plus_sig", "minus_sig", "plus2_sig", "minus2_sig"));
}

TEST_F(InterpreterTest, WrongPipelineName) {
  create_options_->model_name = "linear_model";
  create_options_->model_version = 0;
  create_options_->model_path =
      devtools_build::GetDataDependencyFilepath(kLinearModelPath);
  create_options_->execute_worker_thread_pool =
      execute_worker_thread_pool_.get();
  create_options_->ifrt_client = client_;
  create_options_->computation_thread_pool = computation_thread_pool_.get();
  TF_ASSERT_OK_AND_ASSIGN(auto interpreter,
                          Interpreter::Create(std::move(create_options_)));
  std::vector<Value> inputs = {Value(AsHostBuffer<float>(
      {{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}}, xla::ifrt::Shape({4, 2})))};
  EXPECT_THAT(interpreter->Run(/*pipeline_name=*/"do_not_exist", inputs),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Pipeline name \"do_not_exist\" not found")));
}

TEST_F(InterpreterTest, GetPipelineSignature) {
  create_options_->model_name = "make_number_from_digits_dict";
  create_options_->model_version = 0;
  create_options_->model_path = devtools_build::GetDataDependencyFilepath(
      kMakeNumberFromDigitsDictModelPath);
  create_options_->execute_worker_thread_pool =
      execute_worker_thread_pool_.get();
  create_options_->ifrt_client = client_;
  create_options_->computation_thread_pool = computation_thread_pool_.get();
  TF_ASSERT_OK_AND_ASSIGN(auto interpreter,
                          Interpreter::Create(std::move(create_options_)));
  ASSERT_OK_AND_ASSIGN(auto signature, interpreter->GetPipelineSignature(
                                           "make_number_from_digits"));
  {
    const std::vector<MetaArgSpec> expected_input_specs = {
        MetaArgSpec{
            .spec =
                ArgSpec{
                    .dtype = xla::ifrt::DType(xla::ifrt::DType::kS32),
                    .shape = ArgShape({4, 2}),
                    .sharding = xla::HloSharding::IotaTile({2, 2}),
                },
            .name = "fifth_digit",
        },
        MetaArgSpec{
            .spec =
                ArgSpec{
                    .dtype = xla::ifrt::DType(xla::ifrt::DType::kS32),
                    .shape = ArgShape({4, 2}),
                    .sharding = xla::HloSharding::IotaTile({2, 2}),
                },
            .name = "fourth_digit",
        },
        MetaArgSpec{
            .spec =
                ArgSpec{
                    .dtype = xla::ifrt::DType(xla::ifrt::DType::kS32),
                    .shape = ArgShape({4, 2}),
                    .sharding = xla::HloSharding::IotaTile({2, 2}),
                },
            .name = "third_digit",
        }};
    EXPECT_THAT(signature.input_signature,
                MetaArgSpecsAre(expected_input_specs));
  }
  {
    const std::vector<MetaArgSpec> expected_output_specs = {
        MetaArgSpec{
            .spec = ArgSpec{.dtype = xla::ifrt::DType(xla::ifrt::DType::kS32),
                            .shape = ArgShape({4, 2}),
                            .sharding = xla::HloSharding::Replicate()},
            .name = "y1"},
        MetaArgSpec{
            .spec = ArgSpec{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                            .shape = ArgShape({4, 2}),
                            .sharding = xla::HloSharding::Replicate()},
            .name = "y2"},
    };
    EXPECT_THAT(signature.output_signature,
                MetaArgSpecsAre(expected_output_specs));
  }
}

TEST_F(InterpreterTest, GetPipelineSignatureWithoutNames) {
  create_options_->model_name = "x_times_y";
  create_options_->model_version = 0;
  create_options_->model_path =
      devtools_build::GetDataDependencyFilepath(kXTimesYModelPath);
  create_options_->execute_worker_thread_pool =
      execute_worker_thread_pool_.get();
  create_options_->ifrt_client = client_;
  create_options_->computation_thread_pool = computation_thread_pool_.get();
  TF_ASSERT_OK_AND_ASSIGN(auto interpreter,
                          Interpreter::Create(std::move(create_options_)));
  ASSERT_OK_AND_ASSIGN(auto signature,
                       interpreter->GetPipelineSignature("x_times_y"));
  {
    const std::vector<MetaArgSpec> expected_input_specs = {
        MetaArgSpec{
            .spec = ArgSpec{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                            .shape = ArgShape({4, 2}),
                            .sharding = xla::HloSharding::Replicate()},
        },
        MetaArgSpec{
            .spec = ArgSpec{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                            .shape = ArgShape({2, 2}),
                            .sharding = xla::HloSharding::Replicate()},
        },
    };
    EXPECT_THAT(signature.input_signature,
                MetaArgSpecsAre(expected_input_specs));
  }
  {
    const std::vector<MetaArgSpec> expected_output_specs = {
        MetaArgSpec{
            .spec = ArgSpec{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                            .shape = ArgShape({4, 2}),
                            .sharding = xla::HloSharding::Replicate()},
        },
    };
    EXPECT_THAT(signature.output_signature,
                MetaArgSpecsAre(expected_output_specs));
  }
}

TEST_F(InterpreterTest, LoadGraphFromPipelineComponents) {
  create_options_->lazy_loading_pipelines = true;
  create_options_->model_name = "linear_model";
  create_options_->model_version = 0;
  create_options_->model_path =
      devtools_build::GetDataDependencyFilepath(kLinearModelPath);
  create_options_->execute_worker_thread_pool =
      execute_worker_thread_pool_.get();
  create_options_->ifrt_client = client_;
  create_options_->computation_thread_pool = computation_thread_pool_.get();
  TF_ASSERT_OK_AND_ASSIGN(auto interpreter,
                          Interpreter::Create(std::move(create_options_)));
  orbax_export_orchestration::Pipeline pipeline;
  auto* component = pipeline.add_components();
  component->set_function_name("linear");
  component->set_role(orbax_export_orchestration::ROLE_MODEL);
  component->set_weights_name("params");
  ASSERT_OK_AND_ASSIGN(KernelGraph graph,
                       InterpreterPeer::LoadGraph(interpreter.get(), pipeline));
  EXPECT_EQ(graph.size(), 1);
}

TEST_F(InterpreterTest, LoadGraphWithoutPipelineComponents) {
  create_options_->lazy_loading_pipelines = true;
  create_options_->model_name = "linear_model";
  create_options_->model_version = 0;
  create_options_->model_path =
      devtools_build::GetDataDependencyFilepath(kLinearModelPath);
  create_options_->execute_worker_thread_pool =
      execute_worker_thread_pool_.get();
  create_options_->ifrt_client = client_;
  create_options_->computation_thread_pool = computation_thread_pool_.get();
  TF_ASSERT_OK_AND_ASSIGN(auto interpreter,
                          Interpreter::Create(std::move(create_options_)));
  orbax_export_orchestration::Pipeline pipeline;
  auto* model_function = pipeline.add_model_functions();
  model_function->set_model_function_name("linear");
  model_function->set_weights_name("params");
  ASSERT_OK_AND_ASSIGN(KernelGraph graph,
                       InterpreterPeer::LoadGraph(interpreter.get(), pipeline));
  EXPECT_EQ(graph.size(), 1);
}

///////////////////////////////////////////
///// InterpreterFullyInitializedTest /////
///////////////////////////////////////////

struct InterpreterFullyInitializedTestParam {
  absl::string_view model_name;
  absl::string_view model_path;
  absl::string_view pipeline_name;
};

class InterpreterFullyInitializedTest
    : public InterpreterTest,
      public testing::WithParamInterface<InterpreterFullyInitializedTestParam> {
 public:
  void SetUp() override {
    InterpreterTestBase::SetUp();
    auto create_options = std::make_unique<InterpreterCreateOptions>();
    const std::string platform_name_override =
        absl::GetFlag(FLAGS_jsv_testonly_platform_name_override);
    if (!platform_name_override.empty()) {
      create_options->platform_name = platform_name_override;
    }
    create_options->model_name = GetParam().model_name;
    create_options->model_version = 0;
    create_options->model_path =
        devtools_build::GetDataDependencyFilepath(GetParam().model_path);
    create_options->execute_worker_thread_pool =
        execute_worker_thread_pool_.get();
    create_options->ifrt_client = client_;
    create_options->computation_thread_pool = computation_thread_pool_.get();
    TF_ASSERT_OK_AND_ASSIGN(interpreter_,
                            Interpreter::Create(std::move(create_options)));
  }
};

TEST_P(InterpreterFullyInitializedTest, PreviouslyExecutedInputShapesSucceeds) {
  std::vector<Value> inputs = {Value(AsHostBuffer<float>(
      std::vector<float>(2048, 2.0), xla::ifrt::Shape({2, 1024})))};
  std::vector<testing::Matcher<const Value&>> expected_outputs = {
      ValueOfType<HostBuffer>(HostBufferEq(AsHostBuffer<float>(
          {524288.0, 524288.0}, xla::ifrt::Shape({2, 1}))))};
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto outputs,
        interpreter_->Run(GetParam().pipeline_name, absl::MakeSpan(inputs)));
    EXPECT_THAT(outputs, ElementsAreArray(expected_outputs));
  }
  TF_ASSERT_OK(interpreter_->MarkFullyInitialized());
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto outputs,
        interpreter_->Run(GetParam().pipeline_name, absl::MakeSpan(inputs)));
    EXPECT_THAT(outputs, ElementsAreArray(expected_outputs));
  }
}

TEST_P(InterpreterFullyInitializedTest, NewInputShapesFails) {
  std::vector<Value> inputs = {Value(AsHostBuffer<float>(
      std::vector<float>(2048, 2.0), xla::ifrt::Shape({2, 1024})))};
  std::vector<testing::Matcher<const Value&>> expected_outputs = {
      ValueOfType<HostBuffer>(HostBufferEq(AsHostBuffer<float>(
          {524288.0, 524288.0}, xla::ifrt::Shape({2, 1}))))};
  std::vector<Value> new_inputs = {Value(AsHostBuffer<float>(
      std::vector<float>(4096, 2.0), xla::ifrt::Shape({4, 1024})))};
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto outputs,
        interpreter_->Run(GetParam().pipeline_name, absl::MakeSpan(inputs)));
    EXPECT_THAT(outputs, ElementsAreArray(expected_outputs));
  }
  TF_ASSERT_OK(interpreter_->MarkFullyInitialized());
  {
    EXPECT_THAT(
        interpreter_->Run(GetParam().pipeline_name, new_inputs),
        StatusIs(
            absl::StatusCode::kFailedPrecondition,
            HasSubstr(
                "Got new input shapes requiring a new executable, but can't "
                "create it after full initialization. Please warm up the model "
                "with all the input shapes that are possible during serving. "
                "(See go/tf-model-warmup for more details.)")));
  }
}

TEST_P(InterpreterFullyInitializedTest,
       NewInputShapesBeforeFullyInitializedSucceeds) {
  std::vector<Value> inputs = {Value(AsHostBuffer<float>(
      std::vector<float>(2048, 2.0), xla::ifrt::Shape({2, 1024})))};
  std::vector<testing::Matcher<const Value&>> expected_outputs = {
      ValueOfType<HostBuffer>(HostBufferEq(AsHostBuffer<float>(
          {524288.0, 524288.0}, xla::ifrt::Shape({2, 1}))))};
  std::vector<Value> new_inputs = {Value(AsHostBuffer<float>(
      std::vector<float>(4096, 2.0), xla::ifrt::Shape({4, 1024})))};
  std::vector<testing::Matcher<const Value&>> new_expected_outputs = {
      ValueOfType<HostBuffer>(HostBufferEq(
          AsHostBuffer<float>({524288.0, 524288.0, 524288.0, 524288.0},
                              xla::ifrt::Shape({4, 1}))))};
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto outputs,
        interpreter_->Run(GetParam().pipeline_name, absl::MakeSpan(inputs)));
    EXPECT_THAT(outputs, ElementsAreArray(expected_outputs));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(auto outputs,
                            interpreter_->Run(GetParam().pipeline_name,
                                              absl::MakeSpan(new_inputs)));
    EXPECT_THAT(outputs, ElementsAreArray(new_expected_outputs));
  }
  TF_ASSERT_OK(interpreter_->MarkFullyInitialized());
}

INSTANTIATE_TEST_SUITE_P(
    InterpreterFullyInitializedTestInstances, InterpreterFullyInitializedTest,
    testing::ValuesIn<InterpreterFullyInitializedTestParam>({
        {
            // A test model where multiple compilations are possible.
            .model_name = "dynamic_shape",
            .model_path = kDynamicShapePath,
            .pipeline_name = "tiny_dnn",
        },
        {
            // Marking fully initialized (and preventing new compilations)
            // should still be in effect even if compilation is done inside the
            // batched graph.
            .model_name = "dynamic_shape_with_batching",
            .model_path = kDynamicShapeWithBatchingPath,
            .pipeline_name = "tiny_dnn",
        },
    }));

////////////////////////////////
///// InterpreterMapIoTest /////
////////////////////////////////

class InterpreterMapIoTest
    : public InterpreterTest,
      public testing::WithParamInterface<InterpreterMapIoTestParam> {
  void SetUp() override {
    InterpreterTest::SetUp();
    const auto& params = GetParam();
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
    ScopedMockLog log(kDoNotCaptureLogsYet);
    if (GetParam().model_path == kMultipleApplyFnSharingCkptPath) {
      EXPECT_CALL(log, Log).Times(AnyNumber());
      EXPECT_CALL(log, Log(base_logging::INFO, _,
                           HasSubstr("Skipping loading duplicated Weight")))
          .Times(2);
      log.StartCapturingLogs();
    }
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
};

TEST_P(InterpreterMapIoTest, LoadAndRunSucceedsWithMapReturn) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto outputs, interpreter_->RunWithMapReturn(GetParam().pipeline_name,
                                                     /*positional_inputs=*/{},
                                                     GetParam().inputs));
  EXPECT_THAT(outputs, UnorderedElementsAreArray(GetParam().expected_outputs));
}

TEST_P(InterpreterMapIoTest, LoadAndRunAsyncSucceedsWithMapReturn) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto outputs,
      interpreter_
          ->RunAsyncWithMapReturn(GetParam().pipeline_name,
                                  /*positional_inputs=*/{}, GetParam().inputs)
          .Await());
  EXPECT_THAT(outputs, UnorderedElementsAreArray(GetParam().expected_outputs));
}

INSTANTIATE_TEST_SUITE_P(
    InterpreterMapIoTestInstances, InterpreterMapIoTest,
    testing::ValuesIn<InterpreterMapIoTestParam>({
        {.testcase_name = "linear_model_polymorphic",
         .model_path = kLinearModelPolymorphicPath,
         .pipeline_name = "linear",
         .inputs =
             {
                 {"x", Value(AsHostBuffer<float>({{1.0, 2.0, 3.0, 4.0}},
                                                 xla::ifrt::Shape({2, 2})))},
             },
         .expected_outputs =
             {
                 Pair("y",
                      ValueOfType<HostBuffer>(HostBufferEq(AsHostBuffer<float>(
                          {{4.0, 5.0, 8.0, 9.0}}, xla::ifrt::Shape({2, 2}))))),
             }},
        {.testcase_name = "make_number_from_digits_model",
         .model_path = kMakeNumberFromDigitsDictModelPath,
         .pipeline_name = "make_number_from_digits",
         .inputs = {{"third_digit",
                     Value(AsHostBuffer<int32_t>({4, 5, 6, 7, 8, 9, 3, 2},
                                                 xla::ifrt::Shape({4, 2})))},
                    {"fourth_digit",
                     Value(AsHostBuffer<int32_t>({1, 2, 3, 4, 5, 6, 7, 8},
                                                 xla::ifrt::Shape({4, 2})))},
                    {"fifth_digit",
                     Value(AsHostBuffer<int32_t>({9, 8, 7, 6, 5, 4, 3, 2},
                                                 xla::ifrt::Shape({4, 2})))}},
         .expected_outputs =
             {Pair("y1",
                   ValueOfType<HostBuffer>(HostBufferEq(AsHostBuffer<int32_t>(
                       {13419, 24528, 13637, 24746, 13855, 24964, 13373, 24282},
                       xla::ifrt::Shape({4, 2}))))),
              Pair("y2",
                   ValueOfType<HostBuffer>(HostBufferEq(AsHostBuffer<float>(
                       {{0.13419, 0.24528, 0.13637, 0.24746, 0.13855, 0.24964,
                         0.13373, 0.24282}},
                       xla::ifrt::Shape({4, 2})))))}},
        {.testcase_name = "multiple_apply_fn_model_plus",
         .model_path = kMultipleApplyFnSharingCkptPath,
         .pipeline_name = "plus_sig",
         .inputs =
             {
                 {"x", Value(AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                                                  7.0, 8.0},
                                                 xla::ifrt::Shape({4, 2})))},
             },
         .expected_outputs =
             {
                 Pair("y", ValueOfType<HostBuffer>(
                               HostBufferEq(AsHostBuffer<float>(
                                   {2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0},
                                   xla::ifrt::Shape({4, 2}))))),
             }},
        {.testcase_name = "multiple_apply_fn_model_minus",
         .model_path = kMultipleApplyFnSharingCkptPath,
         .pipeline_name = "minus_sig",
         .inputs =
             {
                 {"x", Value(AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                                                  7.0, 8.0},
                                                 xla::ifrt::Shape({4, 2})))},
             },
         .expected_outputs =
             {
                 Pair("y", ValueOfType<HostBuffer>(
                               HostBufferEq(AsHostBuffer<float>(
                                   {0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0,
                                    -7.0},
                                   xla::ifrt::Shape({4, 2}))))),
             }},
        {.testcase_name = "multiple_apply_fn_model_plus2",
         .model_path = kMultipleApplyFnSharingCkptPath,
         .pipeline_name = "plus2_sig",
         .inputs =
             {
                 {"x", Value(AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                                                  7.0, 8.0},
                                                 xla::ifrt::Shape({4, 2})))},
             },
         .expected_outputs =
             {
                 Pair("y", ValueOfType<HostBuffer>(
                               HostBufferEq(AsHostBuffer<float>(
                                   {3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0},
                                   xla::ifrt::Shape({4, 2}))))),
             }},
        {.testcase_name = "multiple_apply_fn_model_minus2",
         .model_path = kMultipleApplyFnSharingCkptPath,
         .pipeline_name = "minus2_sig",
         .inputs =
             {
                 {"x", Value(AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                                                  7.0, 8.0},
                                                 xla::ifrt::Shape({4, 2})))},
             },
         .expected_outputs =
             {
                 Pair("y", ValueOfType<HostBuffer>(
                               HostBufferEq(AsHostBuffer<float>(
                                   {1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0,
                                    -6.0},
                                   xla::ifrt::Shape({4, 2}))))),
             }},
    }));

TEST_F(InterpreterTest, CreateFailsWithNonExistentModel) {
  create_options_->model_name = "test_model";
  create_options_->model_version = 0;
  create_options_->model_path = "non_existent_model_path";
  create_options_->execute_worker_thread_pool =
      execute_worker_thread_pool_.get();
  create_options_->ifrt_client = client_;
  create_options_->computation_thread_pool = computation_thread_pool_.get();
  EXPECT_THAT(Interpreter::Create(std::move(create_options_)),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("orbax_model_version.txt")));
}

TEST_F(InterpreterTest, LoadFailsWithNoOptionalFunctionHandler) {
  if (optional_function_handler_router_
          ->GetHandler(kTfConcreteFunctionHandleMimeType)
          .ok()) {
    GTEST_SKIP() << "Optional function handler is already registered. This "
                    "test is not applicable.";
  }
  create_options_->model_name = "simple_model_with_tf_processing";
  create_options_->model_version = 0;
  create_options_->model_path = devtools_build::GetDataDependencyFilepath(
      kSimpleModelWithTfProcessingPath);
  create_options_->execute_worker_thread_pool =
      execute_worker_thread_pool_.get();
  create_options_->ifrt_client = client_;
  create_options_->computation_thread_pool = computation_thread_pool_.get();
  EXPECT_THAT(
      // The model has pre-processor and post-processor functions, but no
      // optional function handler is registered.
      Interpreter::Create(std::move(create_options_)),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("No `OptionalFunctionHandler` is found for key")));
}

TEST_F(InterpreterTest, RunReturnsCancelledOnDeadlineExceeded) {
  create_options_->model_name = "x_times_y";
  create_options_->model_version = 0;
  create_options_->model_path =
      devtools_build::GetDataDependencyFilepath(kXTimesYModelPath);
  create_options_->execute_worker_thread_pool =
      execute_worker_thread_pool_.get();
  create_options_->ifrt_client = client_;
  create_options_->computation_thread_pool = computation_thread_pool_.get();
  TF_ASSERT_OK_AND_ASSIGN(auto interpreter,
                          Interpreter::Create(std::move(create_options_)));

  const std::vector<Value> inputs = {
      Value(AsHostBuffer<float>({{0.0, 1.0, 2.0, 3.0}},
                                xla::ifrt::Shape({2, 2}))),
      Value(AsHostBuffer<float>({{0.0, 1.0, 2.0, 3.0}},
                                xla::ifrt::Shape({2, 2})))};

  // Running with an expired execute_deadline should result in kCancelled.
  EXPECT_THAT(
      interpreter->Run(/*pipeline_name=*/"x_times_y",
                       /*positional_inputs=*/inputs,
                       /*keyword_inputs=*/{},
                       /*run_options=*/
                       RunOptions{.deadline = absl::Now() - absl::Seconds(1)}),
      StatusIs(absl::StatusCode::kCancelled));
}

TEST_F(InterpreterTest, Metrics) {
  CellReader<int64_t> model_loading_count_reader(
      "/orbax/serving/model_loading_count");
  CellReader<int64_t> model_loaded_count_reader(
      "/orbax/serving/model_loaded_count");
  CellReader<int64_t> request_count_reader("/orbax/serving/request_count");
  std::string model_name = "x_times_y";
  int64_t model_version = 0;
  {
    create_options_->model_name = model_name;
    create_options_->model_version = model_version;
    create_options_->model_path =
        devtools_build::GetDataDependencyFilepath(kXTimesYModelPath);
    create_options_->execute_worker_thread_pool =
        execute_worker_thread_pool_.get();
    create_options_->ifrt_client = client_;
    create_options_->computation_thread_pool = computation_thread_pool_.get();
    TF_ASSERT_OK_AND_ASSIGN(auto interpreter,
                            Interpreter::Create(std::move(create_options_)));

    const std::vector<Value> inputs = {
        Value(AsHostBuffer<float>({{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}},
                                  xla::ifrt::Shape({4, 2}))),
        Value(AsHostBuffer<float>({{0.0, 1.0, 2.0, 3.0}},
                                  xla::ifrt::Shape({2, 2})))};

    EXPECT_OK(interpreter->Run(/*pipeline_name=*/"x_times_y",
                               /*positional_inputs=*/inputs,
                               /*keyword_inputs=*/{},
                               /*run_options=*/RunOptions{}));
  }

  EXPECT_EQ(model_loading_count_reader.Delta(
                model_name, absl::StrCat(model_version),
                absl::StatusCodeToString(absl::StatusCode::kOk)),
            1);
  EXPECT_GE(
      model_loaded_count_reader.Read(model_name, absl::StrCat(model_version)),
      1);
  EXPECT_EQ(request_count_reader.Delta(
                model_name, absl::StrCat(model_version), "sync",
                absl::StatusCodeToString(absl::StatusCode::kOk)),
            1);
}

}  // namespace
}  // namespace jsv
