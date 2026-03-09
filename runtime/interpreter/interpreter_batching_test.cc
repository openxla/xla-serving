#include <cstdint>
#include <cstdlib>
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
#include "third_party/absl/base/no_destructor.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/flags/flag.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/match.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/str_split.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/time/time.h"
#include "third_party/absl/types/span.h"
#include "third_party/neptune/runtime/core/batching/batch_stats.h"
#include "third_party/neptune/runtime/core/host_buffer.h"
#include "third_party/neptune/runtime/core/host_buffer_matcher.h"
#include "third_party/neptune/runtime/core/host_buffer_test_utils.h"
#include "third_party/neptune/runtime/core/value.h"
#include "third_party/neptune/runtime/core/value_matcher.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/shape.h"
#include "third_party/tensorflow/compiler/xla/tsl/concurrency/future.h"
#include "third_party/tensorflow/compiler/xla/tsl/lib/monitoring/cell_reader.h"
#include "third_party/tensorflow/compiler/xla/tsl/lib/monitoring/test_utils.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"
#include "third_party/tensorflow/compiler/xla/xla_data.proto.h"
#include "third_party/tensorflow/core/common_runtime/cost_constants.h"
#include "third_party/tensorflow/core/common_runtime/cost_measurement.h"
#include "third_party/tensorflow/core/common_runtime/cost_measurement_registry.h"
#include "third_party/tensorflow/core/common_runtime/request_cost.h"
#include "third_party/tensorflow/core/common_runtime/request_cost_accessor.h"
#include "third_party/tensorflow/core/common_runtime/request_cost_accessor_registry.h"

namespace jsv {
namespace {
using ::tensorflow::CostMeasurement;
using ::tensorflow::kNoSmearSuffix;
using ::tensorflow::kTpuCostName;
using ::tensorflow::kWithSmearSuffix;
using ::tensorflow::RequestCost;
using ::tensorflow::RequestCostAccessor;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::Pair;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;
using ::tsl::monitoring::testing::CellReader;
using ::tsl::monitoring::testing::Histogram;

const char kTestCostName[] = "test";

constexpr absl::string_view kLinearModelWithBatchingPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/linear_model_with_batching_gen";
constexpr absl::string_view kLinearModelWithBatchingUnshardedPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/linear_model_with_batching_unsharded_gen";
constexpr absl::string_view kLinearModelWithBatchingBatchDownPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/linear_model_with_batching_batch_down_gen";
constexpr absl::string_view kLinearModelWithBatchingMinimizeCostPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/linear_model_with_batching_minimize_cost_gen";

class InterpreterBatchingTestBase : public InterpreterTestBase {
 protected:
  std::unique_ptr<Interpreter> interpreter_;
};

class LinearModelBatchingTest
    : public InterpreterBatchingTestBase,
      public testing::WithParamInterface<absl::string_view> {
 public:
  void SetUp() override {
    InterpreterBatchingTestBase::SetUp();
    TF_ASSERT_OK_AND_ASSIGN(interpreter_,
                            CreateInterpreter(/*enable_batching=*/true));
    executed_batch_size_reader_ = std::make_unique<CellReader<int64_t>>(
        "/orbax/serving/batching/executed_batch_size");
    batch_costs_reader_ = std::make_unique<CellReader<Histogram>>(
        "/orbax/serving/batching/costs");
  }

 protected:
  absl::StatusOr<std::unique_ptr<Interpreter>> CreateInterpreter(
      bool enable_batching) {
    auto create_options = std::make_unique<InterpreterCreateOptions>();
    create_options->min_max_enqueued_batches = 1;
    create_options->enable_batching = enable_batching;
    const std::string platform_name_override =
        absl::GetFlag(FLAGS_jsv_testonly_platform_name_override);
    if (!platform_name_override.empty()) {
      create_options->platform_name = platform_name_override;
    }
    create_options->model_name = model_name_;
    create_options->model_version = 0;
    create_options->model_path =
        devtools_build::GetDataDependencyFilepath(GetParam());
    create_options->execute_worker_thread_pool =
        execute_worker_thread_pool_.get();
    create_options->ifrt_client = client_;
    create_options->computation_thread_pool = computation_thread_pool_.get();
    return Interpreter::Create(std::move(create_options));
  }

  const std::string model_name_ = "linear_model_with_batching";
  absl::string_view pipeline_name_ = "linear";
  std::unique_ptr<CellReader<int64_t>> executed_batch_size_reader_;
  std::unique_ptr<CellReader<Histogram>> batch_costs_reader_ =
      std::make_unique<CellReader<Histogram>>("/orbax/serving/batching/costs");
};

class TestCostMeasurement : public CostMeasurement {
 public:
  using CostMeasurement::CostMeasurement;

  absl::Duration GetTotalCost() override { return absl::Milliseconds(200); }
  absl::string_view GetCostType() const override { return kTestCostName; }
};
REGISTER_COST_MEASUREMENT(kTestCostName, TestCostMeasurement);

class TestRequestCostAccessor : public RequestCostAccessor {
 public:
  RequestCost* GetRequestCost() const override {
    static absl::NoDestructor<RequestCost> request_cost;
    return request_cost.get();
  }
};
REGISTER_REQUEST_COST_ACCESSOR(kTestCostName, TestRequestCostAccessor);

class TestTpuCostMeasurement : public CostMeasurement {
 public:
  using CostMeasurement::CostMeasurement;

  absl::Duration GetTotalCost() override { return absl::Milliseconds(200); }
  absl::string_view GetCostType() const override { return kTpuCostName; }
};
REGISTER_COST_MEASUREMENT(kTpuCostName, TestTpuCostMeasurement);

class TestTpuRequestCostAccessor : public RequestCostAccessor {
 public:
  RequestCost* GetRequestCost() const override {
    static absl::NoDestructor<RequestCost> request_cost;
    return request_cost.get();
  }
};
REGISTER_REQUEST_COST_ACCESSOR(kTpuCostName, TestTpuRequestCostAccessor);

std::vector<std::string> GetCostTypes() {
  std::vector<std::string> cost_measurement_types =
      absl::StrSplit(std::getenv("TF_COST_MEASUREMENT_TYPE"), ',');
  std::vector<std::string> cost_types;
  cost_types.reserve(cost_measurement_types.size() * 2);
  for (const auto& cost_type : cost_measurement_types) {
    cost_types.push_back(absl::StrCat(cost_type, kWithSmearSuffix));
    cost_types.push_back(absl::StrCat(cost_type, kNoSmearSuffix));
  }
  return cost_types;
}

TEST_P(LinearModelBatchingTest, BatchingEnabled) {
  std::vector<Value> inputs_1 = {Value(AsHostBuffer<float>(
      {{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}}, xla::ifrt::Shape({4, 2})))};
  std::vector<Value> inputs_2 = {Value(
      AsHostBuffer<float>({{5.0, 9.0, 10.0, 11.0}}, xla::ifrt::Shape({2, 2})))};
  std::vector<::testing::Matcher<const Value&>> expected_outputs_1 = {
      ValueOfType<HostBuffer>(HostBufferEq(
          AsHostBuffer<float>({{2.0, 3.0, 6.0, 7.0, 10.0, 11.0, 14.0, 15.0}},
                              xla::ifrt::Shape({4, 2}))))};
  std::vector<::testing::Matcher<const Value&>> expected_outputs_2 = {
      ValueOfType<HostBuffer>(HostBufferEq((AsHostBuffer<float>(
          {{15.0, 16.0, 22.0, 23.0}}, xla::ifrt::Shape({2, 2})))))};
  int num_requests = 16;
  std::vector<std::vector<Value>> inputs;
  std::vector<std::vector<::testing::Matcher<const Value&>>> expected_outputs;
  for (int i = 0; i < num_requests; ++i) {
    if (i % 2 == 0) {
      inputs.push_back(inputs_1);
      expected_outputs.push_back(expected_outputs_1);
    } else {
      inputs.push_back(inputs_2);
      expected_outputs.push_back(expected_outputs_2);
    }
  }
  std::vector<tsl::Future<std::vector<Value>>> futures;
  for (int i = 0; i < num_requests; ++i) {
    futures.push_back(interpreter_->RunAsync(pipeline_name_, inputs[i]));
  }
  for (int i = 0; i < num_requests; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(std::vector<Value> outputs, futures[i].Await());
    EXPECT_THAT(outputs, ElementsAreArray(expected_outputs[i]));
  }
  // The largest allowed batch size is 16. There are a total number of 16
  // requests, 8 requests with batch size 2 and 8 requests with batch size 4. So
  // with large batch splitting enabled, the batch size 16 should be executed 3
  // times ((8 * 2 + 8 * 4) / 16).
  EXPECT_EQ(executed_batch_size_reader_->Delta(model_name_,
                                               /*model_version=*/"0", "16"),
            3);
  for (int batch_size : {2, 4, 8}) {
    EXPECT_EQ(executed_batch_size_reader_->Delta(model_name_,
                                                 /*model_version=*/"0",
                                                 absl::StrCat(batch_size)),
              0);
  }

  // Test the batching costs are recorded correctly.
  Histogram batching_costs;
  std::vector<std::string> cost_types = GetCostTypes();
  for (const auto& cost_type : cost_types) {
    // We do not know the exact time the requests are executed, but it should be
    // non-zero.
    batching_costs =
        batch_costs_reader_->Delta(model_name_,
                                   /*model_version=*/"0",
                                   /*processed_size=*/"16", cost_type);
    EXPECT_EQ(batching_costs.num(), 3);
    EXPECT_GT(batching_costs.sum(), 0);
    for (int batch_size : {2, 4, 8}) {
      batching_costs = batch_costs_reader_->Delta(
          model_name_,
          /*model_version=*/"0",
          /*processed_size=*/absl::StrCat(batch_size), cost_type);
      EXPECT_EQ(batching_costs.sum(), 0);
    }
  }
}

TEST_P(LinearModelBatchingTest, BatchingDisabled) {
  TF_ASSERT_OK_AND_ASSIGN(interpreter_,
                          CreateInterpreter(/*enable_batching=*/false));
  std::vector<Value> inputs_1 = {Value(AsHostBuffer<float>(
      {{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}}, xla::ifrt::Shape({4, 2})))};
  std::vector<Value> inputs_2 = {Value(
      AsHostBuffer<float>({{5.0, 9.0, 10.0, 11.0}}, xla::ifrt::Shape({2, 2})))};
  std::vector<::testing::Matcher<const Value&>> expected_outputs_1 = {
      ValueOfType<HostBuffer>(HostBufferEq(
          AsHostBuffer<float>({{2.0, 3.0, 6.0, 7.0, 10.0, 11.0, 14.0, 15.0}},
                              xla::ifrt::Shape({4, 2}))))};
  std::vector<::testing::Matcher<const Value&>> expected_outputs_2 = {
      ValueOfType<HostBuffer>(HostBufferEq((AsHostBuffer<float>(
          {{15.0, 16.0, 22.0, 23.0}}, xla::ifrt::Shape({2, 2})))))};
  int num_requests = 16;
  std::vector<std::vector<Value>> inputs;
  std::vector<std::vector<::testing::Matcher<const Value&>>> expected_outputs;
  for (int i = 0; i < num_requests; ++i) {
    if (i % 2 == 0) {
      inputs.push_back(inputs_1);
      expected_outputs.push_back(expected_outputs_1);
    } else {
      inputs.push_back(inputs_2);
      expected_outputs.push_back(expected_outputs_2);
    }
  }
  std::vector<tsl::Future<std::vector<Value>>> futures;
  for (int i = 0; i < num_requests; ++i) {
    futures.push_back(interpreter_->RunAsync(pipeline_name_, inputs[i]));
  }
  for (int i = 0; i < num_requests; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(std::vector<Value> outputs, futures[i].Await());
    EXPECT_THAT(outputs, ElementsAreArray(expected_outputs[i]));
  }

  // With batching disabled, no batching-specific metrics should be recorded.
  for (int batch_size : {2, 4, 8, 16}) {
    EXPECT_EQ(executed_batch_size_reader_->Delta(model_name_,
                                                 /*model_version=*/"0",
                                                 absl::StrCat(batch_size)),
              0);
  }

  // Test the batching costs are not recorded.
  Histogram batching_costs;
  std::vector<std::string> cost_types = GetCostTypes();
  for (const auto& cost_type : cost_types) {
    for (int batch_size : {2, 4, 8, 16}) {
      batching_costs = batch_costs_reader_->Delta(
          model_name_, /*model_version=*/"0",
          /*processed_size=*/absl::StrCat(batch_size), cost_type);
      EXPECT_EQ(batching_costs.sum(), 0);
    }
  }
}

TEST_P(LinearModelBatchingTest,
       RequestsWithLargerThanMaxAllowedBatchSizeSucceeds) {
  // Two requests have batch size larger than the largest allowed batch size 16.
  std::vector<int> batch_sizes = {16, 32, 24};
  std::vector<std::vector<Value>> inputs;
  inputs.reserve(batch_sizes.size());
  std::vector<::testing::Matcher<const Value&>> expected_outputs;
  expected_outputs.reserve(batch_sizes.size());
  for (int idx = 0; idx < batch_sizes.size(); ++idx) {
    int batch_size = batch_sizes[idx];
    std::vector<float> input_data(batch_size * 2, idx);
    inputs.push_back({Value(
        AsHostBuffer<float>(input_data, xla::ifrt::Shape({batch_size, 2})))});
    std::vector<float> output_data(batch_size * 2);
    for (int i = 0; i < output_data.size(); ++i) {
      // weights is [[1, 1], [1, 1]] and bias is [1, 2]
      if (i % 2 == 0) {
        output_data[i] = idx * 2 + 1;
      } else {
        output_data[i] = idx * 2 + 2;
      }
    }
    expected_outputs.push_back(ValueOfType<HostBuffer>(HostBufferEq(
        AsHostBuffer<float>(output_data, xla::ifrt::Shape({batch_size, 2})))));
  }
  std::vector<tsl::Future<std::vector<Value>>> futures;
  for (int i = 0; i < batch_sizes.size(); ++i) {
    futures.push_back(interpreter_->RunAsync(pipeline_name_, inputs[i]));
  }
  for (int i = 0; i < batch_sizes.size(); ++i) {
    TF_ASSERT_OK_AND_ASSIGN(std::vector<Value> outputs, futures[i].Await());
    EXPECT_THAT(outputs, ElementsAre(expected_outputs[i]));
  }
  // The largest allowed batch size is 16. The request with batch size 16 will
  // be executed directly. The request with batch size 32 will be split into two
  // requests with batch size 16. The request with batch size 24 will be
  // split into two requests with batch size 16 and 8.
  EXPECT_EQ(executed_batch_size_reader_->Delta(model_name_,
                                               /*model_version=*/"0", "16"),
            4);
  EXPECT_EQ(executed_batch_size_reader_->Delta(model_name_,
                                               /*model_version=*/"0", "8"),
            1);
  for (int batch_size : {2, 4}) {
    EXPECT_EQ(executed_batch_size_reader_->Delta(model_name_,
                                                 /*model_version=*/"0",
                                                 absl::StrCat(batch_size)),
              0);
  }
}

TEST_P(LinearModelBatchingTest, RequestsWithLargerThanMaxBatchSizeFails) {
  // The maximum batch size is 32. A request with a larger batch size
  // should fail.
  int batch_size_exceeding_max = 33;
  EXPECT_THAT(
      interpreter_
          ->RunAsync(pipeline_name_,
                     {Value(AsHostBuffer<float>(
                         std::vector<float>(batch_size_exceeding_max * 2, 1.0),
                         xla::ifrt::Shape({batch_size_exceeding_max, 2})))})
          .Await(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("larger than maximum input batch size")));
}

TEST_P(LinearModelBatchingTest, PadUp) {
  // Send a request with batch size 1, which will be padded up to batch size 2.
  tsl::Future<absl::flat_hash_map<std::string, Value>> future =
      interpreter_->RunAsyncWithMapReturn(
          pipeline_name_, {},
          {{"x", Value(AsHostBuffer<float>({{0.0, 1.0}},
                                           xla::ifrt::Shape({1, 2})))}});
  EXPECT_THAT(future.Await(),
              IsOkAndHolds(ElementsAre(Pair(
                  "y", ValueOfType<HostBuffer>(HostBufferEq(AsHostBuffer<float>(
                           {{2.0, 3.0}}, xla::ifrt::Shape({1, 2}))))))));
  EXPECT_EQ(executed_batch_size_reader_->Delta(model_name_,
                                               /*model_version=*/"0", "2"),
            1);
  for (int batch_size : {4, 8, 16}) {
    EXPECT_EQ(executed_batch_size_reader_->Delta(model_name_,
                                                 /*model_version=*/"0",
                                                 absl::StrCat(batch_size)),
              0);
  }

  // Test the batching costs are recorded correctly.
  Histogram batching_costs;
  std::vector<std::string> cost_types = GetCostTypes();
  ASSERT_EQ(cost_types.size(), 2);
  for (const auto& cost_type : cost_types) {
    for (int batch_size : {4, 8, 16}) {
      batching_costs = batch_costs_reader_->Delta(
          model_name_,
          /*model_version=*/"0",
          /*processed_size=*/absl::StrCat(batch_size), cost_type);
      EXPECT_EQ(batching_costs.sum(), 0);
    }
  }
  Histogram test_with_smear_cost =
      batch_costs_reader_->Delta(model_name_,
                                 /*model_version=*/"0",
                                 /*processed_size=*/"2", cost_types[0]);
  Histogram test_no_smear_cost =
      batch_costs_reader_->Delta(model_name_,
                                 /*model_version=*/"0",
                                 /*processed_size=*/"2", cost_types[1]);
  EXPECT_EQ(test_with_smear_cost.num(), 1);
  EXPECT_EQ(test_no_smear_cost.num(), 1);
  EXPECT_GT(test_with_smear_cost.sum(), 0);
  // Actual batch size is 1, but with padding, the processed size is 2.
  EXPECT_EQ(test_with_smear_cost.sum(), 2 * test_no_smear_cost.sum());
}

INSTANTIATE_TEST_SUITE_P(
    LinearModelBatchingTest, LinearModelBatchingTest,
    testing::Values(kLinearModelWithBatchingPath,
                    kLinearModelWithBatchingUnshardedPath));

class LinearModelBatchPaddingPolicyTest
    : public InterpreterBatchingTestBase,
      public testing::WithParamInterface<absl::string_view> {
 public:
  void SetUp() override {
    InterpreterBatchingTestBase::SetUp();

    executed_batch_size_reader_ = std::make_unique<CellReader<int64_t>>(
        "/orbax/serving/batching/executed_batch_size");
    padding_sizes_reader_ = std::make_unique<CellReader<Histogram>>(
        "/orbax/serving/batching/padding_sizes");
  }

  absl::StatusOr<std::unique_ptr<Interpreter>> CreateInterpreter(
      absl::string_view model_path) {
    auto create_options = std::make_unique<InterpreterCreateOptions>();
    std::string platform_name_override =
        absl::GetFlag(FLAGS_jsv_testonly_platform_name_override);
    if (!platform_name_override.empty()) {
      create_options->platform_name = std::move(platform_name_override);
    }
    create_options->model_name = model_name_;
    create_options->model_version = 0;
    create_options->model_path =
        devtools_build::GetDataDependencyFilepath(model_path);
    create_options->execute_worker_thread_pool =
        execute_worker_thread_pool_.get();
    create_options->ifrt_client = client_;
    create_options->computation_thread_pool = computation_thread_pool_.get();
    return Interpreter::Create(std::move(create_options));
  }

  const std::string model_name_ = "linear_model_with_batching";
  absl::string_view pipeline_name_ = "linear";
  std::unique_ptr<CellReader<int64_t>> executed_batch_size_reader_;
  std::unique_ptr<CellReader<Histogram>> padding_sizes_reader_;
};

TEST_F(LinearModelBatchPaddingPolicyTest, BatchDownTrimmed) {
  TF_ASSERT_OK_AND_ASSIGN(
      interpreter_, CreateInterpreter(kLinearModelWithBatchingBatchDownPath));
  int num_requests = 3;
  std::vector<Value> input = {Value(
      AsHostBuffer<float>({{0.0, 1.0, 2.0, 3.0}}, xla::ifrt::Shape({2, 2})))};
  std::vector<::testing::Matcher<const Value&>> expected_output = {
      ValueOfType<HostBuffer>(HostBufferEq(AsHostBuffer<float>(
          {{2.0, 3.0, 6.0, 7.0}}, xla::ifrt::Shape({2, 2}))))};

  std::vector<std::vector<Value>> inputs;
  std::vector<std::vector<::testing::Matcher<const Value&>>> expected_outputs;
  for (int i = 0; i < num_requests; ++i) {
    inputs.push_back(input);
    expected_outputs.push_back(expected_output);
  }
  std::vector<tsl::Future<std::vector<Value>>> futures;
  for (int i = 0; i < num_requests; ++i) {
    futures.push_back(interpreter_->RunAsync(pipeline_name_, inputs[i]));
  }
  for (int i = 0; i < num_requests; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(std::vector<Value> outputs, futures[i].Await());
    EXPECT_THAT(outputs, ElementsAreArray(expected_outputs[i]));
  }

  // The allowed batch sizes are {2, 4, 8, 16}. There are 3 requests with
  // total size 6. With batch down padding policy, the batch size is trimmed to
  // the smaller allowed batch size 4. So there are 1 batch with batch size 4
  // and 1 batch with batch size 2. No padding needed.
  for (int batch_size : {2, 4}) {
    EXPECT_EQ(executed_batch_size_reader_->Delta(model_name_,
                                                 /*model_version=*/"0",
                                                 absl::StrCat(batch_size)),
              1);
  }
  for (int batch_size : {8, 16}) {
    EXPECT_EQ(executed_batch_size_reader_->Delta(model_name_,
                                                 /*model_version=*/"0",
                                                 absl::StrCat(batch_size)),
              0);
  }
  Histogram padding_sizes_size_4 =
      padding_sizes_reader_->Delta(model_name_,
                                   /*model_version=*/"0",
                                   /*processed_size=*/"4");
  EXPECT_EQ(padding_sizes_size_4.sum(), 0);
  Histogram padding_sizes_size_2 =
      padding_sizes_reader_->Delta(model_name_,
                                   /*model_version=*/"0",
                                   /*processed_size=*/"2");
  EXPECT_EQ(padding_sizes_size_2.sum(), 0);
}

TEST_F(LinearModelBatchPaddingPolicyTest, BatchDownNotTrimmed) {
  TF_ASSERT_OK_AND_ASSIGN(
      interpreter_, CreateInterpreter(kLinearModelWithBatchingBatchDownPath));
  std::vector<Value> input = {
      Value(AsHostBuffer<float>({0.0, 1.0}, xla::ifrt::Shape({1, 2})))};
  std::vector<::testing::Matcher<const Value&>> expected_output = {
      ValueOfType<HostBuffer>(HostBufferEq(
          AsHostBuffer<float>({2.0, 3.0}, xla::ifrt::Shape({1, 2}))))};

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Value> outputs,
      interpreter_->RunAsync(pipeline_name_, input).Await());
  EXPECT_THAT(outputs, ElementsAreArray(expected_output));

  // The allowed batch sizes are {2, 4, 8, 16}. There is 1 request with size 1.
  // With batch down padding policy, the batch size is not trimmed because it's
  // less than the smallest allowed batch size 2. Therefore the batch size 1 is
  // padded up (padding size is 1) to the next allowed batch size 2.
  EXPECT_EQ(executed_batch_size_reader_->Delta(model_name_,
                                               /*model_version=*/"0", "2"),
            1);
  for (int batch_size : {4, 8, 16}) {
    EXPECT_EQ(executed_batch_size_reader_->Delta(model_name_,
                                                 /*model_version=*/"0",
                                                 absl::StrCat(batch_size)),
              0);
  }
  Histogram padding_sizes_size_2 =
      padding_sizes_reader_->Delta(model_name_,
                                   /*model_version=*/"0",
                                   /*processed_size=*/"2");
  EXPECT_EQ(padding_sizes_size_2.sum(), 1);
  EXPECT_EQ(padding_sizes_size_2.num(2), 1);
}

TEST_F(LinearModelBatchPaddingPolicyTest, MinimizeCostWithBatchDown) {
  TF_ASSERT_OK_AND_ASSIGN(
      interpreter_,
      CreateInterpreter(kLinearModelWithBatchingMinimizeCostPath));
  // Reset the batch stats and register batch stats for batch size 2, 4 and 8.
  ModelBatchStats& model_stats =
      GlobalBatchStatsRegistry().model(model_name_, /*model_version=*/0);
  CostTracker* tracker_2 = &model_stats.batch_size(2).tpu_cost();
  CostTracker* tracker_4 = &model_stats.batch_size(4).tpu_cost();
  CostTracker* tracker_8 = &model_stats.batch_size(8).tpu_cost();
  CostTrackerTestPeer::Reset(tracker_2);
  CostTrackerTestPeer::Reset(tracker_4);
  CostTrackerTestPeer::Reset(tracker_8);
  tracker_4->Register(absl::Milliseconds(100));
  tracker_8->Register(absl::Milliseconds(180));
  // Verify the costs are reset and registered correctly.
  EXPECT_FALSE(tracker_2->mean().has_value());
  EXPECT_EQ(tracker_4->mean(), absl::Milliseconds(100));
  EXPECT_EQ(tracker_8->mean(), absl::Milliseconds(180));


  int num_requests = 3;
  std::vector<Value> input = {Value(
      AsHostBuffer<float>({{0.0, 1.0, 2.0, 3.0}}, xla::ifrt::Shape({2, 2})))};
  std::vector<::testing::Matcher<const Value&>> expected_output = {
      ValueOfType<HostBuffer>(HostBufferEq(AsHostBuffer<float>(
          {{2.0, 3.0, 6.0, 7.0}}, xla::ifrt::Shape({2, 2}))))};
  std::vector<std::vector<Value>> inputs;
  std::vector<std::vector<::testing::Matcher<const Value&>>> expected_outputs;
  for (int i = 0; i < num_requests; ++i) {
    inputs.push_back(input);
    expected_outputs.push_back(expected_output);
  }
  std::vector<tsl::Future<std::vector<Value>>> futures;
  for (int i = 0; i < num_requests; ++i) {
    futures.push_back(interpreter_->RunAsync(pipeline_name_, inputs[i]));
  }
  for (int i = 0; i < num_requests; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(std::vector<Value> outputs, futures[i].Await());
    EXPECT_THAT(outputs, ElementsAreArray(expected_outputs[i]));
  }

  // The allowed batch sizes are {2, 4, 8, 16}. There are 3 requests with
  // total size 6. With MINIMIZE_TPU_COST_PER_REQUEST padding policy, the batch
  // down size is 4 and the batch down cost is 100ms/4 = 25ms. The pad up size
  // is 8 and the pad up cost is 180ms/6 = 30ms. Since the batch down cost is
  // less than the pad up cost, BATCH_DOWN is selected. The batch size is
  // trimmed to the smaller allowed batch size 4. So there are 1 batch with
  // batch size 4 and 1 batch with batch size 2. No padding needed.
  for (int batch_size : {2, 4}) {
    EXPECT_EQ(executed_batch_size_reader_->Delta(model_name_,
                                                 /*model_version=*/"0",
                                                 absl::StrCat(batch_size)),
              1);
  }
  for (int batch_size : {8, 16}) {
    EXPECT_EQ(executed_batch_size_reader_->Delta(model_name_,
                                                 /*model_version=*/"0",
                                                 absl::StrCat(batch_size)),
              0);
  }
  Histogram padding_sizes_size_4 =
      padding_sizes_reader_->Delta(model_name_,
                                   /*model_version=*/"0",
                                   /*processed_size=*/"4");
  EXPECT_EQ(padding_sizes_size_4.sum(), 0);
  Histogram padding_sizes_size_2 =
      padding_sizes_reader_->Delta(model_name_,
                                   /*model_version=*/"0",
                                   /*processed_size=*/"2");
  EXPECT_EQ(padding_sizes_size_2.sum(), 0);

  // During batching process cleanup, if the cost measurement type is TPU, the
  // batching cost of a particular batch size will be registered. In the unit
  // test, the default cost of all batch sizes is set to 200ms, while in real
  // production, the cost of different batch sizes should be different. After
  // the first batch (size 4), the batching cost of batch size 4 is registered
  // as 200ms, therefore the cost mean becomes (100 + 200) / 2 = 150ms. After
  // the second batch (size 2), the batching cost of batch size 2 is registered
  // for the first time, therefore the cost mean is 200ms. No batch with size 8
  // has been processed, so the cost mean is still 180ms for batch size 8.
  if (absl::StrContains(std::getenv("TF_COST_MEASUREMENT_TYPE"),
                        kTpuCostName)) {
    EXPECT_EQ(tracker_2->mean(), absl::Milliseconds(200));
    EXPECT_EQ(tracker_4->mean(), absl::Milliseconds(150));
    EXPECT_EQ(tracker_8->mean(), absl::Milliseconds(180));
  } else {
    // No new batching cost is registered when the cost type is not TPU.
    EXPECT_FALSE(tracker_2->mean().has_value());
    EXPECT_EQ(tracker_4->mean(), absl::Milliseconds(100));
    EXPECT_EQ(tracker_8->mean(), absl::Milliseconds(180));
  }
}

TEST_F(LinearModelBatchPaddingPolicyTest, MinimizeCostWithPadUp) {
  TF_ASSERT_OK_AND_ASSIGN(
      interpreter_,
      CreateInterpreter(kLinearModelWithBatchingMinimizeCostPath));
  // Reset the batch stats and register batch stats for batch size 4 and 8.
  ModelBatchStats& model_stats =
      GlobalBatchStatsRegistry().model(model_name_, /*model_version=*/0);
  CostTracker* tracker_4 = &model_stats.batch_size(4).tpu_cost();
  CostTracker* tracker_8 = &model_stats.batch_size(8).tpu_cost();
  CostTrackerTestPeer::Reset(tracker_4);
  CostTrackerTestPeer::Reset(tracker_8);
  tracker_4->Register(absl::Milliseconds(100));
  tracker_8->Register(absl::Milliseconds(100));
  // Verify the costs are reset and registered correctly.
  EXPECT_EQ(tracker_4->mean(), absl::Milliseconds(100));
  EXPECT_EQ(tracker_8->mean(), absl::Milliseconds(100));


  int num_requests = 3;
  std::vector<Value> input = {Value(
      AsHostBuffer<float>({{0.0, 1.0, 2.0, 3.0}}, xla::ifrt::Shape({2, 2})))};
  std::vector<::testing::Matcher<const Value&>> expected_output = {
      ValueOfType<HostBuffer>(HostBufferEq(AsHostBuffer<float>(
          {{2.0, 3.0, 6.0, 7.0}}, xla::ifrt::Shape({2, 2}))))};
  std::vector<std::vector<Value>> inputs;
  std::vector<std::vector<::testing::Matcher<const Value&>>> expected_outputs;
  for (int i = 0; i < num_requests; ++i) {
    inputs.push_back(input);
    expected_outputs.push_back(expected_output);
  }
  std::vector<tsl::Future<std::vector<Value>>> futures;
  for (int i = 0; i < num_requests; ++i) {
    futures.push_back(interpreter_->RunAsync(pipeline_name_, inputs[i]));
  }
  for (int i = 0; i < num_requests; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(std::vector<Value> outputs, futures[i].Await());
    EXPECT_THAT(outputs, ElementsAreArray(expected_outputs[i]));
  }

  // The allowed batch sizes are {2, 4, 8, 16}. There are 3 requests with
  // total size 6. With MINIMIZE_TPU_COST_PER_REQUEST padding policy, the batch
  // down size is 4 and the batch down cost is 100ms/4 = 25ms. The pad up size
  // is 8 and the pad up cost is 100ms/6 = 16ms. Since the batch down cost is
  // higher than the pad up cost, PAD_UP is selected. Therefore the batch size 6
  // is padded to 8. Padding size is 2.
  for (int batch_size : {2, 4, 16}) {
    EXPECT_EQ(executed_batch_size_reader_->Delta(model_name_,
                                                 /*model_version=*/"0",
                                                 absl::StrCat(batch_size)),
              0);
  }
  EXPECT_EQ(executed_batch_size_reader_->Delta(model_name_,
                                               /*model_version=*/"0", "8"),
            1);
  Histogram padding_sizes_size =
      padding_sizes_reader_->Delta(model_name_,
                                   /*model_version=*/"0",
                                   /*processed_size=*/"8");
  EXPECT_EQ(padding_sizes_size.sum(), 2);
  EXPECT_EQ(padding_sizes_size.num(3), 1);

  // During batching process cleanup, if the cost measurement type is TPU, the
  // batching cost of a particular batch size will be registered. In the unit
  // test, the default cost of all batch sizes is set to 200ms, while in real
  // production, the cost of different batch sizes should be different. After
  // the first batch (size 8), the batching cost of batch size 8 is registered
  // as 200ms, therefore the cost mean becomes (100 + 200) / 2 = 150ms. No batch
  // of size 4 has been processed, so the cost mean is still 100ms.
  if (absl::StrContains(std::getenv("TF_COST_MEASUREMENT_TYPE"),
                        kTpuCostName)) {
    EXPECT_EQ(tracker_4->mean().value(), absl::Milliseconds(100));
    EXPECT_EQ(tracker_8->mean().value(), absl::Milliseconds(150));
  } else {
    // No new batching cost is registered when the cost type is not TPU.
    EXPECT_EQ(tracker_4->mean().value(), absl::Milliseconds(100));
    EXPECT_EQ(tracker_8->mean().value(), absl::Milliseconds(100));
  }
}

}  // namespace
}  // namespace jsv
