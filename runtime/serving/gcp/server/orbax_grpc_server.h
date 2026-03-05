#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_SERVING_GCP_SERVER_ORBAX_GRPC_SERVER_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_SERVING_GCP_SERVER_ORBAX_GRPC_SERVER_H_

#include <memory>
#include <string>
#include <utility>

#include "learning/infra/mira/experimental/orbax_model/interpreter/interpreter.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/grpc/include/grpcpp/server_context.h"
#include "third_party/grpc/include/grpcpp/support/status.h"
#include "third_party/neptune/runtime/serving/gcp/proto/get_model_metadata.proto.h"
#include "third_party/neptune/runtime/serving/gcp/proto/predict.proto.h"
#include "third_party/neptune/runtime/serving/gcp/proto/prediction_service.grpc.pb.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/client.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/threadpool.h"

namespace jsv {
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
}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_SERVING_GCP_SERVER_ORBAX_GRPC_SERVER_H_
