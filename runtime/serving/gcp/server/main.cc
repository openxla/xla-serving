// Main entry point for the Orbax Model Server on GCP.
//
// This file is responsible for setting up the server, loading models, and
// handling requests.
#include <memory>
#include <string>

#include "base/init_google.h"
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "include/grpcpp/security/server_credentials.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/server.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/server_builder.h"  // from @com_github_grpc_grpc
#include "runtime/serving/gcp/auth/utils.h"  // from @neptune
#include "runtime/serving/gcp/server/orbax_grpc_server.h"  // from @neptune
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
  auto service_or = jsv::serving::OrbaxGrpcServer::Create(
      absl::GetFlag(FLAGS_model_name), absl::GetFlag(FLAGS_model_path));
  TF_CHECK_OK(service_or.status());

  grpc::ServerBuilder builder;
  std::string private_key_path = absl::GetFlag(FLAGS_server_private_key_path);
  std::string cert_chain_path = absl::GetFlag(FLAGS_server_cert_chain_path);
  std::string root_certs_path = absl::GetFlag(FLAGS_client_cert_chain_path);
  bool use_insecure_grpc_credentials =
      absl::GetFlag(FLAGS_use_insecure_grpc_credentials);
  absl::StatusOr<std::shared_ptr<grpc::ServerCredentials>> creds =
      jsv::serving::CreateServerCredentials(private_key_path, cert_chain_path,
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
  InitGoogle(argv[0], &argc, &argv, /*remove_flags=*/true);
  RunServer();
  return 0;
}
