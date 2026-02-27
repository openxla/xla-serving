#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/time/time.h"
#include "third_party/absl/types/span.h"
#include "third_party/neptune/runtime/core/batching/batch_stats.h"
#include "third_party/neptune/runtime/core/host_buffer.h"
#include "third_party/neptune/runtime/kernel/batching/batch_resource_base.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/dtype.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/shape.h"
#include "third_party/tensorflow/core/common_runtime/cost_constants.h"
#include "third_party/tensorflow/core/common_runtime/cost_measurement.h"
#include "third_party/tensorflow/core/common_runtime/cost_measurement_registry.h"
#include "third_party/tensorflow/core/common_runtime/request_cost.h"

namespace jsv {
namespace {

using ::tensorflow::CostMeasurement;
using ::tensorflow::CostMeasurementRegistry;
using ::tensorflow::kTpuCostName;
using ::tensorflow::RequestCost;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

class TestTpuCostMeasurement : public CostMeasurement {
 public:
  using CostMeasurement::CostMeasurement;

  absl::Duration GetTotalCost() override { return absl::Milliseconds(100); }
  absl::string_view GetCostType() const override { return "test_tpu"; }
};
REGISTER_COST_MEASUREMENT("test_tpu", TestTpuCostMeasurement);

class TestGcuCostMeasurement : public CostMeasurement {
 public:
  using CostMeasurement::CostMeasurement;

  absl::Duration GetTotalCost() override { return absl::Milliseconds(200); }
  absl::string_view GetCostType() const override { return "test_gcu"; }
};
REGISTER_COST_MEASUREMENT("test_gcu", TestGcuCostMeasurement);

std::unique_ptr<BatchResourceBase::BatchTask> MakeBatchTask(
    const int64_t task_size, RequestCost* request_cost,
    absl::Time start_time = absl::UnixEpoch()) {
  auto task = std::make_unique<BatchResourceBase::BatchTask>();
  task->inputs.push_back(HostBuffer(xla::ifrt::DType(xla::ifrt::DType::kF64),
                                    xla::ifrt::Shape({task_size, 1})));
  task->request_cost = request_cost;
  task->start_time = absl::ToUnixNanos(start_time);
  return task;
}

TEST(SplitBatchCostsAndRecordMetricsTest, SkipOnNoCostMeasurement) {
  BatchResourceBase::BatchT batch;
  RequestCost cost;
  batch.AddTask(MakeBatchTask(/*task_size=*/1, &cost));
  batch.Close();

  std::vector<std::unique_ptr<CostMeasurement>> batch_cost_measurements;
  BatchResourceBase::SplitBatchCostsAndRecordMetrics(
      "model_name", 0, batch_cost_measurements, /*processed_size=*/16, batch);
  EXPECT_TRUE(batch.task(0).request_cost->GetCosts().empty());
  EXPECT_THAT(batch.task(0).request_cost->GetBatchMetrics(),
              ::testing::ElementsAre(::testing::FieldsAre(
                  /*processed_size=*/16, /*input_size=*/1, /*padding_size=*/15,
                  ::testing::IsEmpty())));
}

TEST(SplitBatchCostsAndRecordMetricsTest, SkipOnZeroCost) {
  BatchResourceBase::BatchT batch;
  RequestCost cost;
  batch.AddTask(MakeBatchTask(/*task_size=*/1, &cost));
  batch.Close();

  CostMeasurement::Context context{/*is_per_query=*/false};
  std::vector<std::unique_ptr<CostMeasurement>> batch_cost_measurements;
  batch_cost_measurements.push_back(
      CostMeasurementRegistry::CreateByNameOrNull("no_op", context));
  BatchResourceBase::SplitBatchCostsAndRecordMetrics(
      "model_name", 0, batch_cost_measurements, /*processed_size=*/16, batch);
  EXPECT_TRUE(batch.task(0).request_cost->GetCosts().empty());
  EXPECT_THAT(batch.task(0).request_cost->GetBatchMetrics(),
              ::testing::ElementsAre(::testing::FieldsAre(
                  /*processed_size=*/16, /*input_size=*/1, /*padding_size=*/15,
                  ::testing::IsEmpty())));
}

TEST(SplitBatchCostsAndRecordMetricsTest, SkipOnZeroBatchSize) {
  BatchResourceBase::BatchT batch;
  batch.Close();

  CostMeasurement::Context context{/*is_per_query=*/false};
  std::vector<std::unique_ptr<CostMeasurement>> batch_cost_measurements;
  batch_cost_measurements.push_back(
      CostMeasurementRegistry::CreateByNameOrNull("test_tpu", context));
  BatchResourceBase::SplitBatchCostsAndRecordMetrics(
      "model_name", 0, batch_cost_measurements, /*processed_size=*/0, batch);
}

TEST(SplitBatchCostsAndRecordMetricsTest, SkipOnNoRequestCost) {
  BatchResourceBase::BatchT batch;
  batch.AddTask(MakeBatchTask(/*task_size=*/1, nullptr));
  batch.AddTask(MakeBatchTask(/*task_size=*/9, nullptr));
  batch.Close();

  CostMeasurement::Context context{/*is_per_query=*/false};
  std::vector<std::unique_ptr<CostMeasurement>> batch_cost_measurements;
  batch_cost_measurements.push_back(
      CostMeasurementRegistry::CreateByNameOrNull("test_tpu", context));
  BatchResourceBase::SplitBatchCostsAndRecordMetrics(
      "model_name", 0, batch_cost_measurements, /*processed_size=*/16, batch);

  EXPECT_EQ(batch.task(0).request_cost, nullptr);
  EXPECT_EQ(batch.task(1).request_cost, nullptr);
}

TEST(SplitBatchCostsAndRecordMetricsTest, SplitSingleCostType) {
  BatchResourceBase::BatchT batch;
  RequestCost cost1, cost2;
  batch.AddTask(MakeBatchTask(/*task_size=*/1, &cost1));
  batch.AddTask(MakeBatchTask(/*task_size=*/9, &cost2));
  batch.Close();

  CostMeasurement::Context context{/*is_per_query=*/false};
  std::vector<std::unique_ptr<CostMeasurement>> batch_cost_measurements;
  batch_cost_measurements.push_back(
      CostMeasurementRegistry::CreateByNameOrNull("test_tpu", context));
  BatchResourceBase::SplitBatchCostsAndRecordMetrics(
      "model_name", 0, batch_cost_measurements, /*processed_size=*/20, batch);

  EXPECT_THAT(
      batch.task(0).request_cost->GetCosts(),
      UnorderedElementsAre(Pair("test_tpu_with_smear", absl::Milliseconds(10)),
                           Pair("test_tpu_no_smear", absl::Milliseconds(5))));
  EXPECT_THAT(
      batch.task(0).request_cost->GetBatchMetrics(),
      ::testing::ElementsAre(::testing::FieldsAre(
          /*processed_size=*/20, /*input_size=*/1, /*padding_size=*/10,
          UnorderedElementsAre(Pair("test_tpu", absl::Milliseconds(100))))));
  EXPECT_THAT(
      batch.task(1).request_cost->GetCosts(),
      UnorderedElementsAre(Pair("test_tpu_with_smear", absl::Milliseconds(90)),
                           Pair("test_tpu_no_smear", absl::Milliseconds(45))));
  EXPECT_THAT(
      batch.task(1).request_cost->GetBatchMetrics(),
      ::testing::ElementsAre(::testing::FieldsAre(
          /*processed_size=*/20, /*input_size=*/9, /*padding_size=*/10,
          UnorderedElementsAre(Pair("test_tpu", absl::Milliseconds(100))))));
}

TEST(SplitBatchCostsAndRecordMetricsTest, SplitMultiCostTypes) {
  BatchResourceBase::BatchT batch;
  RequestCost cost1, cost2;
  batch.AddTask(MakeBatchTask(/*task_size=*/1, &cost1));
  batch.AddTask(MakeBatchTask(/*task_size=*/9, &cost2));
  batch.Close();

  CostMeasurement::Context context{/*is_per_query=*/false};
  std::vector<std::unique_ptr<CostMeasurement>> batch_cost_measurements;
  batch_cost_measurements.push_back(
      CostMeasurementRegistry::CreateByNameOrNull("test_tpu", context));
  batch_cost_measurements.push_back(
      CostMeasurementRegistry::CreateByNameOrNull("test_gcu", context));
  BatchResourceBase::SplitBatchCostsAndRecordMetrics(
      "model_name", 0, batch_cost_measurements, /*processed_size=*/20, batch);

  EXPECT_THAT(
      batch.task(0).request_cost->GetCosts(),
      UnorderedElementsAre(Pair("test_tpu_with_smear", absl::Milliseconds(10)),
                           Pair("test_tpu_no_smear", absl::Milliseconds(5)),
                           Pair("test_gcu_with_smear", absl::Milliseconds(20)),
                           Pair("test_gcu_no_smear", absl::Milliseconds(10))));
  EXPECT_THAT(
      batch.task(0).request_cost->GetBatchMetrics(),
      ::testing::ElementsAre(::testing::FieldsAre(
          /*processed_size=*/20, /*input_size=*/1, /*padding_size=*/10,
          UnorderedElementsAre(Pair("test_tpu", absl::Milliseconds(100)),
                               Pair("test_gcu", absl::Milliseconds(200))))));

  EXPECT_THAT(
      batch.task(1).request_cost->GetCosts(),
      UnorderedElementsAre(Pair("test_tpu_with_smear", absl::Milliseconds(90)),
                           Pair("test_tpu_no_smear", absl::Milliseconds(45)),
                           Pair("test_gcu_with_smear", absl::Milliseconds(180)),
                           Pair("test_gcu_no_smear", absl::Milliseconds(90))));
  EXPECT_THAT(
      batch.task(1).request_cost->GetBatchMetrics(),
      ::testing::ElementsAre(::testing::FieldsAre(
          /*processed_size=*/20, /*input_size=*/9, /*padding_size=*/10,
          UnorderedElementsAre(Pair("test_tpu", absl::Milliseconds(100)),
                               Pair("test_gcu", absl::Milliseconds(200))))));
}

TEST(SplitBatchCostsAndRecordMetricsTest, SplitOnlyNonZeroCostTypes) {
  BatchResourceBase::BatchT batch;
  RequestCost cost1, cost2;
  batch.AddTask(MakeBatchTask(/*task_size=*/1, &cost1));
  batch.AddTask(MakeBatchTask(/*task_size=*/9, &cost2));
  batch.Close();

  CostMeasurement::Context context{/*is_per_query=*/false};
  std::vector<std::unique_ptr<CostMeasurement>> batch_cost_measurements;
  batch_cost_measurements.push_back(
      CostMeasurementRegistry::CreateByNameOrNull("no_op", context));
  batch_cost_measurements.push_back(
      CostMeasurementRegistry::CreateByNameOrNull("test_tpu", context));
  BatchResourceBase::SplitBatchCostsAndRecordMetrics(
      "model_name", 0, batch_cost_measurements, /*processed_size=*/20, batch);

  EXPECT_THAT(
      batch.task(0).request_cost->GetCosts(),
      UnorderedElementsAre(Pair("test_tpu_with_smear", absl::Milliseconds(10)),
                           Pair("test_tpu_no_smear", absl::Milliseconds(5))));
  EXPECT_THAT(
      batch.task(0).request_cost->GetBatchMetrics(),
      ::testing::ElementsAre(::testing::FieldsAre(
          /*processed_size=*/20, /*input_size=*/1, /*padding_size=*/10,
          UnorderedElementsAre(Pair("test_tpu", absl::Milliseconds(100))))));

  EXPECT_THAT(
      batch.task(1).request_cost->GetCosts(),
      UnorderedElementsAre(Pair("test_tpu_with_smear", absl::Milliseconds(90)),
                           Pair("test_tpu_no_smear", absl::Milliseconds(45))));
  EXPECT_THAT(
      batch.task(1).request_cost->GetBatchMetrics(),
      ::testing::ElementsAre(::testing::FieldsAre(
          /*processed_size=*/20, /*input_size=*/9, /*padding_size=*/10,
          UnorderedElementsAre(Pair("test_tpu", absl::Milliseconds(100))))));
}

TEST(SplitBatchCostsAndRecordMetricsTest, SplitInputTask) {
  BatchResourceBase::BatchT batch;
  RequestCost cost;
  auto parent_task = MakeBatchTask(/*task_size=*/10, &cost);
  std::vector<std::unique_ptr<BatchResourceBase::BatchTask>> child_tasks;
  int task_size = 10;
  for (int i = 0; i < task_size; ++i) {
    auto child_task = parent_task->CreateSplitTask(i, [](absl::Status) {});
    child_task->inputs.push_back(HostBuffer(
        xla::ifrt::DType(xla::ifrt::DType::kF64), xla::ifrt::Shape({1, 1})));
    batch.AddTask(std::move(child_task));
  }
  batch.Close();

  CostMeasurement::Context context{/*is_per_query=*/false};
  std::vector<std::unique_ptr<CostMeasurement>> batch_cost_measurements;
  batch_cost_measurements.push_back(
      CostMeasurementRegistry::CreateByNameOrNull("test_tpu", context));
  BatchResourceBase::SplitBatchCostsAndRecordMetrics(
      "model_name", 0, batch_cost_measurements, /*processed_size=*/20, batch);

  // Check the child tasks have the correct cost. Because all split tasks's
  // request_cost pointer points to the same RequestCost object, the cost is
  // the same for all tasks.
  for (int i = 0; i < batch.num_tasks(); ++i) {
    EXPECT_THAT(batch.task(i).request_cost->GetCosts(),
                UnorderedElementsAre(
                    Pair("test_tpu_with_smear", absl::Milliseconds(100)),
                    Pair("test_tpu_no_smear", absl::Milliseconds(50))));
  }

  // Check the parent task has the correct cost.
  EXPECT_THAT(
      parent_task->request_cost->GetCosts(),
      UnorderedElementsAre(Pair("test_tpu_with_smear", absl::Milliseconds(100)),
                           Pair("test_tpu_no_smear", absl::Milliseconds(50))));
}

TEST(SplitBatchCostsAndRecordMetricsTest, UpdatesGlobalBatchStats) {
  // Create batch_cost_measurements with one TPU cost.
  class FakeTpuCostMeasurement : public CostMeasurement {
   public:
    using CostMeasurement::CostMeasurement;
    absl::Duration GetTotalCost() override { return absl::Hours(555); }
    absl::string_view GetCostType() const override { return kTpuCostName; }
  };
  CostMeasurement::Context context{/* is_per_query= */ false};
  std::vector<std::unique_ptr<CostMeasurement>> batch_cost_measurements;
  batch_cost_measurements.push_back(
      std::make_unique<FakeTpuCostMeasurement>(context));

  // Create a non-empty batch.
  BatchResourceBase::BatchT batch;
  batch.AddTask(MakeBatchTask(/* task_size= */ 1, nullptr));
  batch.Close();

  // Pick a model name that no other test would pick. This is so that we are
  // sure that the CPU cost for this model name has either never been reported
  // before or, if this test is executed multiple times, has been reported by
  // this only.
  const char kModelName[] = "test_updates_global_batch_stats";

  BatchResourceBase::SplitBatchCostsAndRecordMetrics(
      /* model_name= */ kModelName, /* model_version = */ 0,
      batch_cost_measurements,
      /* processed_size= */ 17, batch);

  EXPECT_EQ(GlobalBatchStatsRegistry()
                .model(/* model_name= */ kModelName, /* model_version = */ 0)
                .batch_size(17)
                .tpu_cost()
                .mean(),
            absl::Hours(555));
}

TEST(SplitBatchCostsAndRecordMetricsTest, GlobalBatchStatsProcessedSize) {
  // Create batch_cost_measurements with one TPU cost.
  class FakeTpuCostMeasurement : public CostMeasurement {
   public:
    using CostMeasurement::CostMeasurement;
    absl::Duration GetTotalCost() override { return absl::Hours(555); }
    absl::string_view GetCostType() const override { return kTpuCostName; }
  };
  CostMeasurement::Context context{/* is_per_query= */ false};
  std::vector<std::unique_ptr<CostMeasurement>> batch_cost_measurements;
  batch_cost_measurements.push_back(
      std::make_unique<FakeTpuCostMeasurement>(context));

  // Create a non-empty batch.
  BatchResourceBase::BatchT batch;
  batch.AddTask(MakeBatchTask(/* task_size= */ 1, nullptr));
  batch.Close();

  // Pick a model name that no other test would pick. This is so that we are
  // sure that the CPU cost for this model name has either never been reported
  // before or, if this test is executed multiple times, has been reported by
  // this only.
  const char kModelName[] = "test_global_batch_stats_processed_size";

  // Get the original cumulative processed size.
  int original_cumulative_processed_size =
      GlobalBatchStatsRegistry()
          .model(/* model_name= */ kModelName, /* model_version = */ 0)
          .cumulative_processed_size();

  BatchResourceBase::SplitBatchCostsAndRecordMetrics(
      /* model_name= */ kModelName, /* model_version = */ 0,
      batch_cost_measurements,
      /* processed_size= */ 17, batch);

  // Expect the cumulative processed size to be updated correctly. Note
  // that even though the batch size is 17, there is only one non-padding task,
  // so the cumulative processed size should be
  // original_cumulative_processed_size + 1.
  EXPECT_EQ(GlobalBatchStatsRegistry()
                .model(/* model_name= */ kModelName, /* model_version = */ 0)
                .cumulative_processed_size(),
            original_cumulative_processed_size + 1);

  // Add a second processed batch with three non-padding tasks and a different
  // total batch size.
  BatchResourceBase::BatchT batch2;
  batch2.AddTask(MakeBatchTask(/* task_size= */ 1, nullptr));
  batch2.AddTask(MakeBatchTask(/* task_size= */ 1, nullptr));
  batch2.AddTask(MakeBatchTask(/* task_size= */ 1, nullptr));
  batch2.Close();
  BatchResourceBase::SplitBatchCostsAndRecordMetrics(
      /* model_name= */ kModelName, /* model_version = */ 0,
      batch_cost_measurements,
      /* processed_size= */ 8, batch2);

  // Expect the cumulative processed size to be updated correctly.
  EXPECT_EQ(GlobalBatchStatsRegistry()
                .model(/* model_name= */ kModelName, /* model_version = */ 0)
                .cumulative_processed_size(),
            original_cumulative_processed_size + 4);
}

}  // namespace
}  // namespace jsv
