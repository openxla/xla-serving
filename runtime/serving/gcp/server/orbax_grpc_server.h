#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_SERVING_GCP_SERVER_ORBAX_GRPC_SERVER_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_SERVING_GCP_SERVER_ORBAX_GRPC_SERVER_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "include/grpcpp/server_context.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/support/status.h"  // from @com_github_grpc_grpc
#include "runtime/interpreter/interpreter.h"  // from @neptune
#include "runtime/serving/gcp/proto/get_model_metadata.pb.h"  // from @neptune
#include "runtime/serving/gcp/proto/predict.pb.h"  // from @neptune
#include "runtime/serving/gcp/proto/prediction_service.grpc.pb.h"  // from @neptune
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla

namespace neptune {
namespace serving {

static constexpr char kDefaultServingSignatureDefKey[] = "serving_default";

class OrbaxGrpcServer final : public jsv::serving::PredictionService::Service {
 public:
  OrbaxGrpcServer(OrbaxGrpcServer&& other) = default;
  OrbaxGrpcServer& operator=(OrbaxGrpcServer&& other) = default;

  OrbaxGrpcServer(const OrbaxGrpcServer&) = delete;
  OrbaxGrpcServer& operator=(const OrbaxGrpcServer&) = delete;

  OrbaxGrpcServer(
      absl::string_view model_name,
      std::unique_ptr<jsv::Interpreter> interpreter,
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

  static absl::StatusOr<std::unique_ptr<OrbaxGrpcServer>> Create(
      absl::string_view model_name, absl::string_view model_path);

  // Handles the GetModelMetadata RPC.
  grpc::Status GetModelMetadata(
      grpc::ServerContext* context,
      const jsv::serving::GetModelMetadataRequest* request,
      jsv::serving::GetModelMetadataResponse* response) final;

  // Handles the Predict RPC.
  grpc::Status Predict(grpc::ServerContext* context,
                       const jsv::serving::PredictRequest* request,
                       jsv::serving::PredictResponse* response) final;

 private:
  static constexpr int kThreadPoolNumThreads = 16;

  std::string model_name_;
  std::shared_ptr<xla::ifrt::Client> client_;
  std::unique_ptr<tsl::thread::ThreadPool> execute_worker_thread_pool_;
  std::unique_ptr<tsl::thread::ThreadPool> computation_thread_pool_;
  std::unique_ptr<tsl::thread::ThreadPool> checkpoint_loader_thread_pool_;
  std::unique_ptr<jsv::Interpreter> interpreter_;
};

}  // namespace serving
}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_SERVING_GCP_SERVER_ORBAX_GRPC_SERVER_H_
