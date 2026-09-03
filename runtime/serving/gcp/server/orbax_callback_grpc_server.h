// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_SERVING_GCP_SERVER_ORBAX_CALLBACK_GRPC_SERVER_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_SERVING_GCP_SERVER_ORBAX_CALLBACK_GRPC_SERVER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "include/grpcpp/server_context.h"  // from @grpc
#include "include/grpcpp/support/server_callback.h"  // from @grpc
#include "protos/grpc/get_model_metadata.pb.h"  // from @neptune
#include "protos/grpc/predict.pb.h"  // from @neptune
#include "protos/grpc/prediction_service.grpc.pb.h"  // from @neptune
#include "runtime/core/value.h"  // from @neptune
#include "runtime/interpreter/interpreter.h"  // from @neptune
#include "runtime/interpreter/run_options.h"  // from @neptune
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/tsl/lib/monitoring/gauge.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla

namespace neptune {
namespace serving {

struct OrbaxCallbackGrpcServerOptions {
  bool enable_lazy_loading = true;
  bool lazy_loading_use_graph_executor = true;
  bool enable_mlrt = true;
  bool allow_xla_cpu = true;
  bool hoist_invariant_ops = true;
  bool sink_in_invariant_ops = true;
  int64_t cost_threshold = 10000;
  bool warmup_server = false;
  int thread_pool_num_threads = 0;
  // Maximum number of concurrent in-flight Predict requests before new ones
  // are rejected with RESOURCE_EXHAUSTED. If negative (default), backpressure
  // is disabled (unbounded queue depth). If 0, defaults to 4x
  // thread_pool_num_threads (or 4x CPU count if that is also unset). If
  // positive, used as the explicit limit.
  int max_predict_queue_depth = -1;
};

class OrbaxCallbackGrpcServer final

    : public neptune::serving::PredictionService::CallbackService {
 public:
  OrbaxCallbackGrpcServer(OrbaxCallbackGrpcServer&& other) = default;
  OrbaxCallbackGrpcServer& operator=(OrbaxCallbackGrpcServer&& other) = default;

  OrbaxCallbackGrpcServer(const OrbaxCallbackGrpcServer&) = delete;
  OrbaxCallbackGrpcServer& operator=(const OrbaxCallbackGrpcServer&) = delete;

  OrbaxCallbackGrpcServer(
      absl::string_view model_name,
      std::unique_ptr<neptune::Interpreter> interpreter,
      std::shared_ptr<xla::ifrt::Client> client,
      std::unique_ptr<tsl::thread::ThreadPool> execute_worker_thread_pool,
      std::unique_ptr<tsl::thread::ThreadPool> computation_thread_pool,
      std::unique_ptr<tsl::thread::ThreadPool> checkpoint_loader_thread_pool,
      int max_predict_queue_depth)
      : model_name_(model_name),
        client_(client),
        execute_worker_thread_pool_(std::move(execute_worker_thread_pool)),
        computation_thread_pool_(std::move(computation_thread_pool)),
        checkpoint_loader_thread_pool_(
            std::move(checkpoint_loader_thread_pool)),
        interpreter_(std::move(interpreter)),
        max_predict_queue_depth_(max_predict_queue_depth),
        in_flight_predict_count_(model_name_) {}

  static absl::StatusOr<std::unique_ptr<OrbaxCallbackGrpcServer>> Create(
      absl::string_view model_name, absl::string_view model_path,
      const OrbaxCallbackGrpcServerOptions& options =
          OrbaxCallbackGrpcServerOptions());

  // Handles the GetModelMetadata RPC.
  grpc::ServerUnaryReactor* GetModelMetadata(
      grpc::CallbackServerContext* context,
      const neptune::serving::GetModelMetadataRequest* request,
      neptune::serving::GetModelMetadataResponse* response) override;

  // Handles the Predict RPC.
  grpc::ServerUnaryReactor* Predict(
      grpc::CallbackServerContext* context,
      const neptune::serving::PredictRequest* request,
      neptune::serving::PredictResponse* response) override;

 private:
  static constexpr int kThreadPoolNumThreads = 16;

  // Thread-safe count of concurrent in-flight Predict requests, used to
  // enforce `max_predict_queue_depth_` and reported live via a
  // `tsl::monitoring::Gauge` (labeled by `model_name`) so the effect of
  // `max_predict_queue_depth_` can be validated directly against loadtest
  // dashboards. `mu_` guards both the count and the gauge update together,
  // so the two never observably disagree: without it, a thread whose
  // increment/decrement is stale (but not yet published) could publish
  // after a later, more-current update and leave the gauge stuck on a wrong
  // value until the next call happens to correct it.
  class InFlightRequestCounter {
   public:
    explicit InFlightRequestCounter(absl::string_view model_name);

    // Returns the count after incrementing.
    int Increment() ABSL_LOCKS_EXCLUDED(mu_);
    void Decrement() ABSL_LOCKS_EXCLUDED(mu_);

   private:
    absl::Mutex mu_;
    int count_ ABSL_GUARDED_BY(mu_) = 0;
    tsl::monitoring::GaugeCell<int64_t>* gauge_cell_;
  };

  std::string model_name_;
  std::shared_ptr<xla::ifrt::Client> client_;
  std::unique_ptr<tsl::thread::ThreadPool> execute_worker_thread_pool_;
  std::unique_ptr<tsl::thread::ThreadPool> computation_thread_pool_;
  std::unique_ptr<tsl::thread::ThreadPool> checkpoint_loader_thread_pool_;
  std::unique_ptr<neptune::Interpreter> interpreter_;
  int max_predict_queue_depth_;
  InFlightRequestCounter in_flight_predict_count_;

  absl::Status Warmup(absl::string_view model_path);

  absl::StatusOr<absl::flat_hash_map<std::string, neptune::Value>>
  RunPredictInternal(const neptune::serving::PredictRequest& request,
                     absl::string_view pipeline_name,
                     const neptune::RunOptions& run_options);
};

}  // namespace serving
}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_SERVING_GCP_SERVER_ORBAX_CALLBACK_GRPC_SERVER_H_
