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

#include "runtime/serving/gcp/server/grpc_status_utils.h"  // from @xla_serving

#include <string>

#include "absl/status/status.h"  // from @abseil-cpp
#include "include/grpcpp/support/status.h"  // from @grpc

namespace xla_serving {
namespace serving {

grpc::Status ToGrpcStatus(const absl::Status& status) {
  if (status.ok()) {
    return grpc::Status::OK;
  }
  grpc::StatusCode code;
  switch (status.code()) {
    case absl::StatusCode::kOk:
      return grpc::Status::OK;
    case absl::StatusCode::kCancelled:
      code = grpc::StatusCode::CANCELLED;
      break;
    case absl::StatusCode::kUnknown:
      code = grpc::StatusCode::UNKNOWN;
      break;
    case absl::StatusCode::kInvalidArgument:
      code = grpc::StatusCode::INVALID_ARGUMENT;
      break;
    case absl::StatusCode::kDeadlineExceeded:
      code = grpc::StatusCode::DEADLINE_EXCEEDED;
      break;
    case absl::StatusCode::kNotFound:
      code = grpc::StatusCode::NOT_FOUND;
      break;
    case absl::StatusCode::kAlreadyExists:
      code = grpc::StatusCode::ALREADY_EXISTS;
      break;
    case absl::StatusCode::kPermissionDenied:
      code = grpc::StatusCode::PERMISSION_DENIED;
      break;
    case absl::StatusCode::kUnauthenticated:
      code = grpc::StatusCode::UNAUTHENTICATED;
      break;
    case absl::StatusCode::kResourceExhausted:
      code = grpc::StatusCode::RESOURCE_EXHAUSTED;
      break;
    case absl::StatusCode::kFailedPrecondition:
      code = grpc::StatusCode::FAILED_PRECONDITION;
      break;
    case absl::StatusCode::kAborted:
      code = grpc::StatusCode::ABORTED;
      break;
    case absl::StatusCode::kOutOfRange:
      code = grpc::StatusCode::OUT_OF_RANGE;
      break;
    case absl::StatusCode::kUnimplemented:
      code = grpc::StatusCode::UNIMPLEMENTED;
      break;
    case absl::StatusCode::kInternal:
      code = grpc::StatusCode::INTERNAL;
      break;
    case absl::StatusCode::kUnavailable:
      code = grpc::StatusCode::UNAVAILABLE;
      break;
    case absl::StatusCode::kDataLoss:
      code = grpc::StatusCode::DATA_LOSS;
      break;
    default:
      // Should not happen, but fallback to INTERNAL for any unhandled codes.
      code = grpc::StatusCode::INTERNAL;
      break;
  }
  return grpc::Status(code, std::string(status.message()));
}

}  // namespace serving
}  // namespace xla_serving
