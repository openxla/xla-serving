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

#ifndef XLA_SERVING_RUNTIME_SERVING_GCP_SERVER_ORBAX_CALLBACK_GRPC_SERVER_H_
#define XLA_SERVING_RUNTIME_SERVING_GCP_SERVER_ORBAX_CALLBACK_GRPC_SERVER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "include/grpcpp/server_context.h"  // from @grpc
#include "include/grpcpp/support/server_callback.h"  // from @grpc
#include "protos/grpc/get_model_metadata.pb.h"  // from @xla_serving
#include "protos/grpc/model_server_config.pb.h"  // from @xla_serving
#include "protos/grpc/predict.pb.h"  // from @xla_serving
#include "protos/grpc/prediction_service.grpc.pb.h"  // from @xla_serving
#include "runtime/core/value.h"  // from @xla_serving
#include "runtime/interpreter/interpreter.h"  // from @xla_serving
#include "runtime/interpreter/run_options.h"  // from @xla_serving
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla

namespace xla_serving {
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
  // Only used when built with ENABLE_TF=1: size of the TFRT work queue
  // (intra- and inter-op threads) that runs TensorFlow preprocessor /
  // postprocessor functions.
  int tfrt_num_threads = 8;
  // CPU only. If true, uses the NanoRt IFRT client instead of the PjRt CPU
  // client.
  bool use_nanort_cpu_client = true;
  // Only used when `use_nanort_cpu_client` is true. If true, NanoRt uses the
  // server's computation (intra-op) thread pool for intra-op parallelism, so
  // no extra compute threads are created. Otherwise NanoRt uses its own
  // process-wide intra-op thread pool.
  bool nanort_share_computation_thread_pool = true;
};

class OrbaxCallbackGrpcServer final

    : public xla_serving::serving::PredictionService::CallbackService {
 public:
  struct ModelEntry {
    std::string name;
    std::string path;
    std::unique_ptr<xla_serving::Interpreter> interpreter;
  };

  OrbaxCallbackGrpcServer(OrbaxCallbackGrpcServer&& other) = default;
  OrbaxCallbackGrpcServer& operator=(OrbaxCallbackGrpcServer&& other) = default;

  OrbaxCallbackGrpcServer(const OrbaxCallbackGrpcServer&) = delete;
  OrbaxCallbackGrpcServer& operator=(const OrbaxCallbackGrpcServer&) = delete;

  OrbaxCallbackGrpcServer(
      absl::flat_hash_map<std::string, std::shared_ptr<ModelEntry>> models,
      std::shared_ptr<xla::ifrt::Client> client,
      std::unique_ptr<tsl::thread::ThreadPool> execute_worker_thread_pool,
      std::unique_ptr<tsl::thread::ThreadPool> computation_thread_pool,
      std::unique_ptr<tsl::thread::ThreadPool> checkpoint_loader_thread_pool);

  // Backward-compatible constructor for a single model.
  OrbaxCallbackGrpcServer(
      absl::string_view model_name,
      std::unique_ptr<xla_serving::Interpreter> interpreter,
      std::shared_ptr<xla::ifrt::Client> client,
      std::unique_ptr<tsl::thread::ThreadPool> execute_worker_thread_pool,
      std::unique_ptr<tsl::thread::ThreadPool> computation_thread_pool,
      std::unique_ptr<tsl::thread::ThreadPool> checkpoint_loader_thread_pool);

  // Creates an OrbaxCallbackGrpcServer from a ModelServerConfig.
  static absl::StatusOr<std::unique_ptr<OrbaxCallbackGrpcServer>> Create(
      const ModelServerConfig& config,
      const OrbaxCallbackGrpcServerOptions& options =
          OrbaxCallbackGrpcServerOptions());

  // Creates an OrbaxCallbackGrpcServer for a single model
  // (backward-compatible).
  static absl::StatusOr<std::unique_ptr<OrbaxCallbackGrpcServer>> Create(
      absl::string_view model_name, absl::string_view model_path,
      const OrbaxCallbackGrpcServerOptions& options =
          OrbaxCallbackGrpcServerOptions());

  // Handles the GetModelMetadata RPC.
  grpc::ServerUnaryReactor* GetModelMetadata(
      grpc::CallbackServerContext* context,
      const xla_serving::serving::GetModelMetadataRequest* request,
      xla_serving::serving::GetModelMetadataResponse* response) override;

  // Handles the Predict RPC.
  grpc::ServerUnaryReactor* Predict(
      grpc::CallbackServerContext* context,
      const xla_serving::serving::PredictRequest* request,
      xla_serving::serving::PredictResponse* response) override;

 private:
  static constexpr int kThreadPoolNumThreads = 16;

  // `client_` must be declared before the thread pools so that it is destroyed
  // after them. Weight loading (`LoadOrbaxCheckpointTask`) runs asynchronously
  // on `checkpoint_loader_thread_pool_` with a raw pointer to the client and
  // may still be in flight when the server is destroyed; the pool destructor
  // waits for it, so the client must still be alive at that point. The NanoRt
  // client may hold a pointer to `computation_thread_pool_` as its intra-op
  // pool; that is safe because it does not use the pool once all executions
  // have finished.
  std::shared_ptr<xla::ifrt::Client> client_;
  std::unique_ptr<tsl::thread::ThreadPool> execute_worker_thread_pool_;
  std::unique_ptr<tsl::thread::ThreadPool> computation_thread_pool_;
  std::unique_ptr<tsl::thread::ThreadPool> checkpoint_loader_thread_pool_;
  absl::flat_hash_map<std::string, std::shared_ptr<ModelEntry>> models_;

  absl::Status Warmup(const ModelEntry& model_entry);

  absl::StatusOr<absl::flat_hash_map<std::string, xla_serving::Value>>
  RunPredictInternal(const ModelEntry& model_entry,
                     const xla_serving::serving::PredictRequest& request,
                     absl::string_view pipeline_name,
                     const xla_serving::RunOptions& run_options);
};

}  // namespace serving
}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_SERVING_GCP_SERVER_ORBAX_CALLBACK_GRPC_SERVER_H_
