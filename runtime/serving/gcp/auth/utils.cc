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

#include "runtime/serving/gcp/auth/utils.h"  // from @neptune

#include <memory>
#include <string>

#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "include/grpc/grpc_security_constants.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/security/credentials.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/security/server_credentials.h"  // from @com_github_grpc_grpc
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla

namespace neptune {
namespace serving {

// Creates gRPC server credentials based on the provided flags.
absl::StatusOr<std::shared_ptr<grpc::ServerCredentials>>
CreateServerCredentials(std::string private_key_path,
                        std::string cert_chain_path,
                        std::string root_certs_path,
                        bool use_insecure_grpc_credentials) {
  if (use_insecure_grpc_credentials) {
    LOG(WARNING) << "Using insecure server credentials. This should not be "
                    "used in production. For testing, LocalServerCredentials "
                    "should be used which restricts connections to localhost.";
    return grpc::
        InsecureServerCredentials();  // NOLINT(grpc_insecure_credential_linter)
  }

  if (!private_key_path.empty() && !cert_chain_path.empty()) {
    LOG(INFO) << "Using file-based SSL/TLS server credentials.";
    std::string private_key_pem;
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        tsl::ReadFileToString(tsl::Env::Default(), private_key_path,
                              &private_key_pem),
        "Failed to read server private key from filepath");

    std::string cert_chain_pem;
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        tsl::ReadFileToString(tsl::Env::Default(), cert_chain_path,
                              &cert_chain_pem),
        "Failed to read server certificate chain from filepath");

    grpc::SslServerCredentialsOptions ssl_opts;
    ssl_opts.pem_key_cert_pairs.push_back({private_key_pem, cert_chain_pem});

    if (!root_certs_path.empty()) {
      LOG(INFO) << "Loading root certificates from: " << root_certs_path;
      TF_RETURN_WITH_CONTEXT_IF_ERROR(
          tsl::ReadFileToString(tsl::Env::Default(), root_certs_path,
                                &ssl_opts.pem_root_certs),
          "Failed to read client certificate chain from filepath");
      // Set ssl_opts.client_cert_request_type if client auth is required.
    }
    return grpc::SslServerCredentials(ssl_opts);
  }

  // Use LocalServerCredentials as a fallback. This will only work for local
  // connections.
  LOG(WARNING)
      << "Using LocalServerCredentials for server credentials. This will "
      << "only work for local connections. If connecting remotely,"
      << "you may notice SOCKET_CLOSED errors.";
  return grpc::experimental::LocalServerCredentials(LOCAL_TCP);
}

// Creates gRPC ChannelCredentials based on flag settings.
absl::StatusOr<std::shared_ptr<grpc::ChannelCredentials>>
CreateChannelCredentials(std::string server_cert_chain_path,
                         bool use_insecure_grpc_credentials) {
  if (use_insecure_grpc_credentials) {
    LOG(WARNING) << "Connecting to server with insecure channel credentials. "
                 << "WARNING: THIS SHOULD NOT BE USED IN PRODUCTION.";
    // clang-format off
    // NOLINTNEXTLINE(whitespace/line_length)
    return grpc::InsecureChannelCredentials();  // NOLINT(grpc_insecure_credential_linter)
    // clang-format on
  }

  grpc::SslCredentialsOptions ssl_opts;
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), server_cert_chain_path,
                            &ssl_opts.pem_root_certs),
      "Failed to read server certificate chain from filepath");
  return grpc::SslCredentials(ssl_opts);
}

}  // namespace serving
}  // namespace neptune
