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

#ifndef XLA_SERVING_RUNTIME_SERVING_GCP_AUTH_UTILS_H_
#define XLA_SERVING_RUNTIME_SERVING_GCP_AUTH_UTILS_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "include/grpcpp/security/credentials.h"  // from @grpc
#include "include/grpcpp/security/server_credentials.h"  // from @grpc

namespace xla_serving {
namespace serving {

// Returns true when the server credential flags select LocalServerCredentials
// (loopback-only TCP transport).
bool UsesLocalServerCredentials(absl::string_view private_key_path,
                                absl::string_view cert_chain_path,
                                absl::string_view root_certs_path,
                                bool use_insecure_grpc_credentials,
                                bool use_alts_credentials = false);

// Creates gRPC server credentials based on the provided flags.
absl::StatusOr<std::shared_ptr<grpc::ServerCredentials>>
CreateServerCredentials(std::string private_key_path,
                        std::string cert_chain_path,
                        std::string root_certs_path,
                        bool use_insecure_grpc_credentials,
                        bool use_alts_credentials = false);

// Creates gRPC channel credentials based on the provided flags.
absl::StatusOr<std::shared_ptr<grpc::ChannelCredentials>>
CreateChannelCredentials(std::string server_cert_chain_path,
                         bool use_insecure_grpc_credentials,
                         bool use_alts_credentials = false);

}  // namespace serving
}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_SERVING_GCP_AUTH_UTILS_H_
