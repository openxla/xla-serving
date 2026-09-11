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

// A simple client for the JSV gRPC nnx model server.
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @abseil-cpp
#include "absl/flags/parse.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "include/grpcpp/channel.h"  // from @grpc
#include "include/grpcpp/client_context.h"  // from @grpc
#include "include/grpcpp/create_channel.h"  // from @grpc
#include "include/grpcpp/security/credentials.h"  // from @grpc
#include "include/grpcpp/support/status.h"  // from @grpc
#include "protos/grpc/host_buffer.pb.h"  // from @neptune
#include "protos/grpc/model.pb.h"  // from @neptune
#include "protos/grpc/predict.pb.h"  // from @neptune
#include "protos/grpc/prediction_log.pb.h"  // from @neptune
#include "protos/grpc/prediction_service.grpc.pb.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune
#include "runtime/serving/gcp/auth/utils.h"  // from @neptune
#include "xla/python/ifrt/dtype.pb.h"  // from @xla
#include "xla/python/ifrt/shape.pb.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla

ABSL_FLAG(std::string, server_address, "", "Server IP address and port.");
ABSL_FLAG(std::string, model_name, "", "Name of the model to query.");

ABSL_FLAG(std::string, server_cert_chain_path, "",
          "Path to the server's certificate chain file for SSL/TLS. For a "
          "client, this is the server's certificate chain for verification. If "
          "not provided, the client will fall back on default trust stores.");
ABSL_FLAG(bool, use_insecure_grpc_credentials, false,
          "Whether to use insecure credentials for the server. WARNING: THIS "
          "SHOULD NOT BE USED IN PRODUCTION.");

ABSL_FLAG(int, num_request_features, 2, "Number of request features.");
ABSL_FLAG(int, num_candidate_features, 2, "Number of candidate features.");

void LogPredictResponse(const xla_serving::serving::PredictResponse& response) {
  LOG(INFO) << "Model Spec: " << response.model_spec().name()
            << " Signature: " << response.model_spec().signature_name();
  LOG(INFO) << "Outputs:";
  if (response.outputs().empty()) {
    LOG(INFO) << "  (No outputs received)";
  } else {
    for (const auto& pair : response.outputs()) {
      const std::string& output_name = pair.first;
      const xla_serving::HostBufferProto& output_buffer = pair.second;
      LOG(INFO) << "  Output: " << output_name << ", DType: "
                << xla::ifrt::DTypeProto::Kind_Name(
                       output_buffer.dtype().kind())
                << ", Shape: ["
                << absl::StrJoin(output_buffer.shape().dims(), ", ") << "]";
      absl::StatusOr<xla_serving::HostBuffer> host_buffer =
          xla_serving::HostBuffer::FromProto(output_buffer);
      if (!host_buffer.ok()) {
        LOG(ERROR) << "Failed to convert HostBufferProto to HostBuffer: "
                   << host_buffer.status();
        continue;
      }
      LOG(INFO) << "  Output data: " << host_buffer->DebugString();
    }
  }
}

absl::StatusOr<xla_serving::serving::PredictResponse> RunPredict(
    const std::string& server_address, absl::string_view model_name,
    std::string server_cert_chain_path, bool use_insecure_grpc_credentials) {
  xla_serving::serving::PredictRequest request;
  request.mutable_model_spec()->set_name(model_name);

  // Construct the prediction request with synthetic data.
  auto* inputs = request.mutable_inputs();
  std::vector<std::string> feature_names;
  for (int i = 1; i <= absl::GetFlag(FLAGS_num_request_features); ++i) {
    feature_names.push_back(absl::StrCat("request_feature_", i));
  }
  for (int i = 1; i <= absl::GetFlag(FLAGS_num_candidate_features); ++i) {
    feature_names.push_back(absl::StrCat("candidate_feature_", i));
  }
  for (const std::string& feature_name : feature_names) {
    xla_serving::HostBufferProto buffer_proto;
    buffer_proto.mutable_dtype()->set_kind(xla::ifrt::DTypeProto::KIND_S32);
    auto* shape = buffer_proto.mutable_shape();
    shape->add_dims(1);
    shape->add_dims(1);

    int32_t zero = 0;
    buffer_proto.set_raw_data(
        std::string(reinterpret_cast<const char*>(&zero), sizeof(zero)));
    (*inputs)[feature_name] = std::move(buffer_proto);
  }
  LOG(INFO) << "Successfully constructed PredictRequest with synthetic data.";
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<grpc::ChannelCredentials> creds,
      xla_serving::serving::CreateChannelCredentials(server_cert_chain_path,
                                             use_insecure_grpc_credentials));
  std::shared_ptr<grpc::Channel> channel =
      grpc::CreateChannel(server_address, creds);

  std::unique_ptr<xla_serving::serving::PredictionService::Stub> stub =
      xla_serving::serving::PredictionService::NewStub(channel);

  xla_serving::serving::PredictResponse response;
  grpc::ClientContext context;
  LOG(INFO) << "Sending PredictRequest to " << server_address;
  grpc::Status predict_status = stub->Predict(&context, request, &response);
  if (!predict_status.ok()) {
    return absl::InternalError(
        absl::StrCat("Predict RPC failed: ", predict_status.error_code(), ": ",
                     predict_status.error_message()));
  }
  return response;
}

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  std::string server_address = absl::GetFlag(FLAGS_server_address);
  std::string model_name = absl::GetFlag(FLAGS_model_name);

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

  auto predict_response =
      RunPredict(server_address, model_name, server_cert_chain_path,
                 use_insecure_grpc_credentials);
  if (!predict_response.ok()) {
    LOG(ERROR) << "Failed to run predict: " << predict_response.status();
    return 1;
  }
  LogPredictResponse(*predict_response);
  return 0;
}
