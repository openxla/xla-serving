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

#include "runtime/serving/gcp/auth/utils.h"  // from @xla_serving

#include <memory>
#include <string>

#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "include/grpc/grpc_security_constants.h"  // from @grpc
#include "include/grpcpp/security/credentials.h"  // from @grpc
#include "include/grpcpp/security/server_credentials.h"  // from @grpc
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla

namespace xla_serving {
namespace serving {

bool UsesLocalServerCredentials(absl::string_view private_key_path,
                                absl::string_view cert_chain_path,
                                absl::string_view root_certs_path,
                                bool use_insecure_grpc_credentials,
                                bool use_alts_credentials) {
  return private_key_path.empty() && cert_chain_path.empty() &&
         root_certs_path.empty() && !use_insecure_grpc_credentials &&
         !use_alts_credentials;
}

// Creates gRPC server credentials based on the provided flags.
absl::StatusOr<std::shared_ptr<grpc::ServerCredentials>>
CreateServerCredentials(std::string private_key_path,
                        std::string cert_chain_path,
                        std::string root_certs_path,
                        bool use_insecure_grpc_credentials,
                        bool use_alts_credentials) {
  const bool has_tls_files = !private_key_path.empty() ||
                             !cert_chain_path.empty() ||
                             !root_certs_path.empty();
  const int modes = (use_insecure_grpc_credentials ? 1 : 0) +
                    (use_alts_credentials ? 1 : 0) + (has_tls_files ? 1 : 0);
  if (modes > 1) {
    return absl::InvalidArgumentError(
        "Cannot combine --use_insecure_grpc_credentials, "
        "--use_alts_credentials, and SSL/TLS certificate flags.");
  }

  if (use_insecure_grpc_credentials) {
    LOG(WARNING) << "Using insecure server credentials. This should not be "
                    "used in production. For testing, LocalServerCredentials "
                    "should be used which restricts connections to localhost.";
    return grpc::
        InsecureServerCredentials();  // NOLINT(grpc_insecure_credential_linter)
  }

  if (use_alts_credentials) {
    LOG(INFO) << "Using ALTS server credentials.";
    // AltsServerCredentials wraps a null C credential outside GCP instead of
    // returning nullptr, whereas AltsCredentials returns nullptr when
    // grpc_alts_is_running_on_gcp() is false. Probe AltsCredentials first so
    // non-GCP environments fail immediately before loading/compiling models.
    if (grpc::experimental::AltsCredentials(
            grpc::experimental::AltsCredentialsOptions()) == nullptr) {
      return absl::FailedPreconditionError(
          "Failed to create ALTS server credentials; ALTS is only supported "
          "when running on Google Cloud (GCE/GKE).");
    }
    return grpc::experimental::AltsServerCredentials(
        grpc::experimental::AltsServerCredentialsOptions());
  }

  if (has_tls_files) {
    if (private_key_path.empty() || cert_chain_path.empty()) {
      return absl::InvalidArgumentError(
          "Both --server_private_key_path and --server_cert_chain_path must be "
          "provided when configuring SSL/TLS server credentials.");
    }
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

    grpc::SslServerCredentialsOptions ssl_opts(
        !root_certs_path.empty()
            ? GRPC_SSL_REQUEST_AND_REQUIRE_CLIENT_CERTIFICATE_AND_VERIFY
            : GRPC_SSL_DONT_REQUEST_CLIENT_CERTIFICATE);
    ssl_opts.pem_key_cert_pairs.push_back({private_key_pem, cert_chain_pem});

    if (!root_certs_path.empty()) {
      LOG(INFO) << "Loading root certificates from: " << root_certs_path;
      TF_RETURN_WITH_CONTEXT_IF_ERROR(
          tsl::ReadFileToString(tsl::Env::Default(), root_certs_path,
                                &ssl_opts.pem_root_certs),
          "Failed to read client certificate chain from filepath");
    }
    return grpc::SslServerCredentials(ssl_opts);
  }

  // Default to LocalServerCredentials, which restricts connections to local
  // loopback interfaces without requiring TLS certificates.
  LOG(INFO) << "Using LocalServerCredentials (LOCAL_TCP); accepting loopback "
            << "connections only. For remote connections, use "
            << "--use_alts_credentials on GCP or configure SSL/TLS certificate "
            << "flags.";
  return grpc::experimental::LocalServerCredentials(LOCAL_TCP);
}

// Creates gRPC ChannelCredentials based on flag settings.
absl::StatusOr<std::shared_ptr<grpc::ChannelCredentials>>
CreateChannelCredentials(std::string server_cert_chain_path,
                         bool use_insecure_grpc_credentials,
                         bool use_alts_credentials) {
  const bool has_tls_cert = !server_cert_chain_path.empty();
  const int modes = (use_insecure_grpc_credentials ? 1 : 0) +
                    (use_alts_credentials ? 1 : 0) + (has_tls_cert ? 1 : 0);
  if (modes > 1) {
    return absl::InvalidArgumentError(
        "Cannot combine --use_insecure_grpc_credentials, "
        "--use_alts_credentials, and --server_cert_chain_path.");
  }

  if (use_insecure_grpc_credentials) {
    LOG(WARNING) << "Connecting to server with insecure channel credentials. "
                 << "WARNING: THIS SHOULD NOT BE USED IN PRODUCTION.";
    // clang-format off
    // NOLINTNEXTLINE(whitespace/line_length)
    return grpc::InsecureChannelCredentials();  // NOLINT(grpc_insecure_credential_linter)
    // clang-format on
  }

  if (use_alts_credentials) {
    LOG(INFO) << "Connecting to server with ALTS channel credentials.";
    std::shared_ptr<grpc::ChannelCredentials> creds =
        grpc::experimental::AltsCredentials(
            grpc::experimental::AltsCredentialsOptions());
    if (creds == nullptr) {
      return absl::FailedPreconditionError(
          "Failed to create ALTS channel credentials; ALTS is only supported "
          "when running on Google Cloud (GCE/GKE).");
    }
    return creds;
  }

  if (has_tls_cert) {
    grpc::SslCredentialsOptions ssl_opts;
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        tsl::ReadFileToString(tsl::Env::Default(), server_cert_chain_path,
                              &ssl_opts.pem_root_certs),
        "Failed to read server certificate chain from filepath");
    return grpc::SslCredentials(ssl_opts);
  }

  LOG(INFO) << "Connecting to server with LocalCredentials (LOCAL_TCP).";
  return grpc::experimental::LocalCredentials(LOCAL_TCP);
}

}  // namespace serving
}  // namespace xla_serving
