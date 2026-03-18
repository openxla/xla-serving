#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_SERVING_GCP_AUTH_UTILS_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_SERVING_GCP_AUTH_UTILS_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "include/grpcpp/security/credentials.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/security/server_credentials.h"  // from @com_github_grpc_grpc

namespace jsv {
namespace serving {

// Creates gRPC server credentials based on the provided flags.
absl::StatusOr<std::shared_ptr<grpc::ServerCredentials>>
CreateServerCredentials(std::string private_key_path,
                        std::string cert_chain_path,
                        std::string root_certs_path,
                        bool use_insecure_grpc_credentials);

// Creates gRPC channel credentials based on the provided flags.
absl::StatusOr<std::shared_ptr<grpc::ChannelCredentials>>
CreateChannelCredentials(std::string server_cert_chain_path,
                         bool use_insecure_grpc_credentials);

}  // namespace serving
}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_SERVING_GCP_AUTH_UTILS_H_
