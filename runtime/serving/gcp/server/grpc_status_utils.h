#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_SERVING_GCP_SERVER_GRPC_STATUS_UTILS_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_SERVING_GCP_SERVER_GRPC_STATUS_UTILS_H_

#include "absl/status/status.h"  // from @abseil-cpp
#include "include/grpcpp/support/status.h"  // from @com_github_grpc_grpc

namespace neptune {
namespace serving {

grpc::Status ToGrpcStatus(const absl::Status& status);

}  // namespace serving
}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_SERVING_GCP_SERVER_GRPC_STATUS_UTILS_H_
