// A simple client for the JSV gRPC server which reads a provided TFRecord file.
#include <memory>
#include <string>

#include "base/init_google.h"
#include "learning/infra/mira/experimental/orbax_model/tensorflow/utils/tf_to_jsv_proto.h"
#include "absl/flags/flag.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "include/grpcpp/channel.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/client_context.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/create_channel.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/security/credentials.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/support/status.h"  // from @com_github_grpc_grpc
#include "runtime/core/host_buffer.h"  // from @neptune
#include "runtime/core/proto/host_buffer.pb.h"  // from @neptune
#include "runtime/serving/gcp/auth/utils.h"  // from @neptune
#include "runtime/serving/gcp/proto/model.pb.h"  // from @neptune
#include "runtime/serving/gcp/proto/predict.pb.h"  // from @neptune
#include "runtime/serving/gcp/proto/prediction_service.grpc.pb.h"  // from @neptune
#include "xla/python/ifrt/dtype.pb.h"  // from @xla
#include "xla/python/ifrt/shape.pb.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "third_party/tensorflow/core/lib/io/record_reader.h"
#include "third_party/tensorflow/core/platform/env.h"
#include "third_party/tensorflow/core/platform/file_system.h"
#include "third_party/tensorflow/core/platform/tstring.h"
#include "third_party/tensorflow_serving/apis/prediction_log.pb.h"

ABSL_FLAG(std::string, server_address, "", "Server IP address and port.");
ABSL_FLAG(std::string, model_name, "", "Name of the model to query.");
ABSL_FLAG(std::string, warmup_requests_path, "",
          "Path to a TFRecord file containing a single PredictRequest proto.");
ABSL_FLAG(std::string, server_cert_chain_path, "",
          "Path to the server's certificate chain file for SSL/TLS. For a "
          "client, this is the server's certificate chain for verification. If "
          "not provided, the client will fall back on default trust stores.");
ABSL_FLAG(bool, use_insecure_grpc_credentials, false,
          "Whether to use insecure credentials for the server. WARNING: THIS "
          "SHOULD NOT BE USED IN PRODUCTION.");

void LogPredictResponse(const jsv::serving::PredictResponse& response) {
  LOG(INFO) << "Model Spec: " << response.model_spec().name()
            << " Signature: " << response.model_spec().signature_name();
  LOG(INFO) << "Outputs:";
  if (response.outputs().empty()) {
    LOG(INFO) << "  (No outputs received)";
  } else {
    for (const auto& pair : response.outputs()) {
      const std::string& output_name = pair.first;
      const jsv::HostBufferProto& output_buffer = pair.second;
      LOG(INFO) << "  Output: " << output_name << ", DType: "
                << xla::ifrt::DTypeProto::Kind_Name(
                       output_buffer.dtype().kind())
                << ", Shape: ["
                << absl::StrJoin(output_buffer.shape().dims(), ", ") << "]";
      absl::StatusOr<jsv::HostBuffer> host_buffer =
          jsv::HostBuffer::FromProto(output_buffer);
      if (!host_buffer.ok()) {
        LOG(ERROR) << "Failed to convert HostBufferProto to HostBuffer: "
                   << host_buffer.status();
        continue;
      }
      LOG(INFO) << "  Output data: "
                << host_buffer->host_array()->DebugString();
    }
  }
}

absl::StatusOr<jsv::serving::PredictResponse> RunPredict(
    const std::string& server_address, absl::string_view model_name,
    const std::string& warmup_requests_path, std::string server_cert_chain_path,
    bool use_insecure_grpc_credentials) {
  // Read PredictRequest from TFRecord
  std::unique_ptr<tensorflow::RandomAccessFile> warmup_requests_file;
  absl::Status status = tensorflow::Env::Default()->NewRandomAccessFile(
      warmup_requests_path, &warmup_requests_file);
  if (!status.ok()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to open TFRecord file: ", warmup_requests_path));
  }

  tensorflow::io::SequentialRecordReader reader(warmup_requests_file.get());
  tensorflow::tstring record;
  status = reader.ReadRecord(&record);
  if (!status.ok()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Failed to read record from TFRecord file: ", warmup_requests_path));
  }
  tensorflow::serving::PredictionLog prediction_log;
  if (!prediction_log.ParseFromString(record)) {
    return absl::InvalidArgumentError(
        "Failed to parse PredictionLog from TFRecord record.");
  }
  tensorflow::serving::PredictRequest tf_request =
      prediction_log.predict_log().request();
  LOG(INFO) << "Successfully loaded PredictRequest from "
            << warmup_requests_path;
  absl::StatusOr<jsv::serving::PredictRequest> request =
      jsv::FromTfPredictRequest(tf_request);
  if (!request.ok()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to convert PredictRequest to JSV proto: ",
                     request.status().message()));
  }
  request->mutable_model_spec()->set_name(model_name);
  LOG(INFO) << "Successfully converted PredictRequest to JSV proto.";
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<grpc::ChannelCredentials> creds,
      jsv::serving::CreateChannelCredentials(server_cert_chain_path,
                                             use_insecure_grpc_credentials));
  std::shared_ptr<grpc::Channel> channel =
      grpc::CreateChannel(server_address, creds);

  std::unique_ptr<jsv::serving::PredictionService::Stub> stub =
      jsv::serving::PredictionService::NewStub(channel);

  jsv::serving::PredictResponse response;
  grpc::ClientContext context;
  LOG(INFO) << "Sending PredictRequest to " << server_address;
  grpc::Status predict_status =
      stub->Predict(&context, request.value(), &response);
  if (!predict_status.ok()) {
    return absl::InternalError(
        absl::StrCat("Predict RPC failed: ", predict_status.error_code(), ": ",
                     predict_status.error_message()));
  }
  return response;
}

int main(int argc, char* argv[]) {
  InitGoogle(argv[0], &argc, &argv, /*remove_flags=*/true);
  std::string server_address = absl::GetFlag(FLAGS_server_address);
  std::string model_name = absl::GetFlag(FLAGS_model_name);
  std::string warmup_requests_path = absl::GetFlag(FLAGS_warmup_requests_path);
  std::string server_cert_chain_path =
      absl::GetFlag(FLAGS_server_cert_chain_path);
  bool use_insecure_grpc_credentials =
      absl::GetFlag(FLAGS_use_insecure_grpc_credentials);

  if (server_address.empty()) {
    LOG(ERROR) << "Server IP address must be provided.";
    return 1;
  }
  if (model_name.empty()) {
    LOG(ERROR) << "Model name must be provided.";
    return 1;
  }
  if (warmup_requests_path.empty()) {
    LOG(ERROR) << "A path to a TFRecord file containing a single "
               << "PredictRequest proto must be provided.";
    return 1;
  }

  auto predict_response =
      RunPredict(server_address, model_name, warmup_requests_path,
                 server_cert_chain_path, use_insecure_grpc_credentials);
  if (!predict_response.ok()) {
    LOG(ERROR) << "Failed to run predict: " << predict_response.status();
    return 1;
  }
  LogPredictResponse(*predict_response);
  return 0;
}
