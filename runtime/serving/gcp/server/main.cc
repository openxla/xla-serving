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

// Main entry point for the Orbax Model Server on GCP.
//
// This file is responsible for setting up the server, loading models, and
// handling requests.
#include <memory>
#include <string>
#include "absl/flags/parse.h"  // from @abseil-cpp
#include "absl/flags/flag.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "include/grpcpp/security/server_credentials.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/server.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/server_builder.h"  // from @com_github_grpc_grpc
#include "runtime/serving/gcp/auth/utils.h"  // from @neptune
#include "runtime/serving/gcp/server/orbax_callback_grpc_server.h"  // from @neptune
#include "xla/tsl/platform/status.h"  // from @xla

ABSL_FLAG(int, port, 8500, "Port for the gRPC server to listen on.");

ABSL_FLAG(std::string, model_name, "", "Name of the model.");
ABSL_FLAG(std::string, model_path, "", "Path to the model.");

ABSL_FLAG(std::string, server_private_key_path, "",
          "Path to the server's private key file for SSL/TLS.");
ABSL_FLAG(std::string, server_cert_chain_path, "",
          "Path to the server's certificate chain file for SSL/TLS. For a "
          "client, this is the server's certificate chain for verification. If "
          "not provided, the client will fall back on default trust stores.");
ABSL_FLAG(
    std::string, client_cert_chain_path, "",
    "Optional path to the certificate chain file for client authentication.");
ABSL_FLAG(bool, use_insecure_grpc_credentials, false,
          "Whether to use insecure credentials for the server. WARNING: THIS "
          "SHOULD NOT BE USED IN PRODUCTION.");

void RunServer() {
  std::string server_address = absl::StrCat("[::]:", absl::GetFlag(FLAGS_port));
  auto service_or = neptune::serving::OrbaxCallbackGrpcServer::Create(
      absl::GetFlag(FLAGS_model_name), absl::GetFlag(FLAGS_model_path));
  TF_CHECK_OK(service_or.status());

  grpc::ServerBuilder builder;
  std::string private_key_path = absl::GetFlag(FLAGS_server_private_key_path);
  std::string cert_chain_path = absl::GetFlag(FLAGS_server_cert_chain_path);
  std::string root_certs_path = absl::GetFlag(FLAGS_client_cert_chain_path);
  bool use_insecure_grpc_credentials =
      absl::GetFlag(FLAGS_use_insecure_grpc_credentials);
  absl::StatusOr<std::shared_ptr<grpc::ServerCredentials>> creds =
      neptune::serving::CreateServerCredentials(private_key_path, cert_chain_path,
                                            root_certs_path,
                                            use_insecure_grpc_credentials);
  TF_CHECK_OK(creds.status());

  builder.AddListeningPort(server_address, *creds);
  builder.RegisterService(service_or.value().get());
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  LOG(INFO) << "JAX Serving listening on " << server_address;

  server->Wait();
}

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  RunServer();
  return 0;
}
