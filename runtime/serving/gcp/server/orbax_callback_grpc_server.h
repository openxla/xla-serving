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

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "include/grpcpp/server_context.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/support/server_callback.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/support/status.h"  // from @com_github_grpc_grpc
#include "runtime/interpreter/interpreter.h"  // from @neptune
#include "runtime/serving/gcp/proto/get_model_metadata.pb.h"  // from @neptune
#include "runtime/serving/gcp/proto/predict.pb.h"  // from @neptune
#include "runtime/serving/gcp/proto/prediction_service.grpc.pb.h"  // from @neptune
#include "xla/python/ifrt/client.h"  // from @xla
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
      std::unique_ptr<tsl::thread::ThreadPool> checkpoint_loader_thread_pool)
      : model_name_(model_name),
        client_(client),
        execute_worker_thread_pool_(std::move(execute_worker_thread_pool)),
        computation_thread_pool_(std::move(computation_thread_pool)),
        checkpoint_loader_thread_pool_(
            std::move(checkpoint_loader_thread_pool)),
        interpreter_(std::move(interpreter)) {}

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

  std::string model_name_;
  std::shared_ptr<xla::ifrt::Client> client_;
  std::unique_ptr<tsl::thread::ThreadPool> execute_worker_thread_pool_;
  std::unique_ptr<tsl::thread::ThreadPool> computation_thread_pool_;
  std::unique_ptr<tsl::thread::ThreadPool> checkpoint_loader_thread_pool_;
  std::unique_ptr<neptune::Interpreter> interpreter_;
};

}  // namespace serving
}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_SERVING_GCP_SERVER_ORBAX_CALLBACK_GRPC_SERVER_H_
