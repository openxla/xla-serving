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

// A simple client for the Neptune gRPC server serving a ViT model.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <memory>
#include <numeric>
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
#include "absl/time/clock.h"  // from @abseil-cpp
#include "absl/time/time.h"  // from @abseil-cpp
#include "include/grpcpp/channel.h"  // from @grpc
#include "include/grpcpp/client_context.h"  // from @grpc
#include "include/grpcpp/create_channel.h"  // from @grpc
#include "include/grpcpp/security/credentials.h"  // from @grpc
#include "include/grpcpp/support/channel_arguments.h"  // from @grpc
#include "include/grpcpp/support/status.h"  // from @grpc
#include "protos/grpc/host_buffer.pb.h"  // from @neptune
#include "protos/grpc/model.pb.h"  // from @neptune
#include "protos/grpc/predict.pb.h"  // from @neptune
#include "protos/grpc/prediction_service.grpc.pb.h"  // from @neptune
#include "runtime/serving/gcp/auth/utils.h"  // from @neptune
#include "xla/python/ifrt/dtype.pb.h"  // from @xla
#include "xla/python/ifrt/shape.pb.h"  // from @xla

ABSL_FLAG(std::string, server_address, "", "Server IP address and port.");
ABSL_FLAG(std::string, model_name, "", "Name of the model to query.");
ABSL_FLAG(int32_t, batch_size, 1, "Batch size for the input image.");
ABSL_FLAG(int32_t, top_k, 5, "Number of top predictions to display.");
ABSL_FLAG(std::string, labels_file, "",
          "Path to a text file with one ImageNet class label per line.");
ABSL_FLAG(std::string, image_file, "",
          "Path to a raw float32 binary file (224*224*3 floats). "
          "Use preprocess_image.py to generate this from a JPEG/PNG.");
ABSL_FLAG(int32_t, num_requests, 1,
          "Number of inference requests to send. Reports latency statistics "
          "when > 1.");
ABSL_FLAG(int32_t, warmup_requests, 1,
          "Number of warmup requests to discard before benchmarking.");
ABSL_FLAG(std::string, server_cert_chain_path, "",
          "Path to the server's certificate chain file for SSL/TLS. For a "
          "client, this is the server's certificate chain for verification. If "
          "not provided, the client will fall back on default trust stores.");
ABSL_FLAG(bool, use_insecure_grpc_credentials, false,
          "Whether to use insecure credentials for the server. WARNING: THIS "
          "SHOULD NOT BE USED IN PRODUCTION.");
ABSL_FLAG(bool, verbose, false,
          "Log per-request status and RPC latency. Set to true to see "
          "per-iteration timing during benchmarking.");

std::vector<std::string> LoadLabels(const std::string& path) {
  std::vector<std::string> labels;
  if (path.empty()) return labels;
  std::ifstream file(path);
  if (!file.is_open()) {
    LOG(WARNING) << "Could not open labels file: " << path;
    return labels;
  }
  std::string line;
  while (std::getline(file, line)) {
    labels.push_back(line);
  }
  return labels;
}

void LogPredictResponse(const xla_serving::serving::PredictResponse& response,
                        int32_t top_k,
                        const std::vector<std::string>& labels) {
  LOG(INFO) << "Model: " << response.model_spec().name()
            << ", Signature: " << response.model_spec().signature_name();

  auto it = response.outputs().find("logits");
  if (it == response.outputs().end()) {
    LOG(ERROR) << "No 'logits' output found in response.";
    return;
  }

  const xla_serving::HostBufferProto& output_buffer = it->second;
  LOG(INFO) << "Output shape: ["
            << absl::StrJoin(output_buffer.shape().dims(), ", ") << "]";

  const auto& raw = output_buffer.raw_data();
  const auto dtype_kind = output_buffer.dtype().kind();
  if (dtype_kind != xla::ifrt::DTypeProto::KIND_F32) {
    LOG(ERROR) << "Expected fp32 logits from server but got dtype kind="
               << dtype_kind
               << ". Re-export the model with fp32 output (cast the "
               << "final logits to float32 in the apply_fn).";
    return;
  }
  if (raw.empty()) {
    LOG(ERROR) << "Logits raw_data is empty.";
    return;
  }
  if (raw.size() % sizeof(float) != 0) {
    LOG(ERROR) << "Logits raw_data size (" << raw.size()
               << ") is not a multiple of sizeof(float).";
    return;
  }
  int num_classes = raw.size() / sizeof(float);
  const float* logits = reinterpret_cast<const float*>(raw.data());

  int k = std::min(std::max(top_k, 0), num_classes);
  absl::Time sort_start = absl::Now();
  std::vector<int> indices(num_classes);
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                    [&](int a, int b) { return logits[a] > logits[b]; });
  absl::Duration sort_duration = absl::Now() - sort_start;

  absl::Time label_start = absl::Now();
  LOG(INFO) << "Top-" << k << " predictions:";
  for (int i = 0; i < k; ++i) {
    int idx = indices[i];
    std::string label =
        (idx < static_cast<int>(labels.size())) ? labels[idx]
                                                : absl::StrCat("class_", idx);
    LOG(INFO) << "  " << (i + 1) << ". [" << idx << "] " << label
              << "  (logit: " << logits[idx] << ")";
  }
  absl::Duration label_duration = absl::Now() - label_start;

  LOG(INFO) << "[Benchmark] Top-" << k << " sort: "
            << absl::ToDoubleMicroseconds(sort_duration) << " us";
  LOG(INFO) << "[Benchmark] Label lookup + print: "
            << absl::ToDoubleMicroseconds(label_duration) << " us";
}

absl::StatusOr<xla_serving::serving::PredictResponse> RunPredict(
    xla_serving::serving::PredictionService::Stub* stub,
    absl::string_view model_name, int32_t batch_size,
    const std::string& image_file, bool verbose) {
  xla_serving::serving::PredictRequest request;
  request.mutable_model_spec()->set_name(model_name);

  auto* inputs = request.mutable_inputs();
  xla_serving::HostBufferProto buffer_proto;
  buffer_proto.mutable_dtype()->set_kind(xla::ifrt::DTypeProto::KIND_F32);
  auto* shape = buffer_proto.mutable_shape();
  shape->add_dims(batch_size);
  shape->add_dims(224);
  shape->add_dims(224);
  shape->add_dims(3);

  size_t num_elements = static_cast<size_t>(batch_size) * 224 * 224 * 3;
  size_t expected_bytes = num_elements * sizeof(float);

  if (!image_file.empty()) {
    std::ifstream file(image_file, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Could not open image file: ", image_file));
    }
    size_t file_size = file.tellg();
    size_t single_image_bytes = 224 * 224 * 3 * sizeof(float);
    if (file_size != single_image_bytes && file_size != expected_bytes) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Image file size mismatch: expected ", single_image_bytes,
          " bytes (single image) or ", expected_bytes,
          " bytes (full batch) but got ", file_size));
    }
    file.seekg(0);
    std::string image_data(file_size, '\0');
    file.read(image_data.data(), file_size);
    if (!file) {
      return absl::InternalError(absl::StrCat(
          "Short read from image file: ", image_file,
          " (expected ", file_size, " bytes)"));
    }
    if (file_size == single_image_bytes && batch_size > 1) {
      std::string raw_data;
      raw_data.reserve(expected_bytes);
      for (int i = 0; i < batch_size; ++i) {
        raw_data.append(image_data);
      }
      buffer_proto.set_raw_data(std::move(raw_data));
    } else {
      buffer_proto.set_raw_data(std::move(image_data));
    }
  } else {
    // float 0.0 is bit-pattern all-zeros, so a zero-filled string is
    // bit-identical to `std::vector<float>(num_elements, 0.0f)`.
    buffer_proto.set_raw_data(std::string(expected_bytes, '\0'));
  }
  (*inputs)["images"] = std::move(buffer_proto);

  if (verbose) {
    LOG(INFO) << "Successfully constructed PredictRequest with synthetic image "
              << "[" << batch_size << ", 224, 224, 3] float32.";
  }

  xla_serving::serving::PredictResponse response;
  grpc::ClientContext context;
  if (verbose) {
    LOG(INFO) << "Sending PredictRequest.";
  }
  absl::Time rpc_start = absl::Now();
  grpc::Status predict_status = stub->Predict(&context, request, &response);
  absl::Duration rpc_duration = absl::Now() - rpc_start;
  if (!predict_status.ok()) {
    return absl::InternalError(
        absl::StrCat("Predict RPC failed: ", predict_status.error_code(), ": ",
                     predict_status.error_message()));
  }
  if (verbose) {
    LOG(INFO) << "[Benchmark] Predict RPC (server inference): "
              << absl::ToDoubleMilliseconds(rpc_duration) << " ms";
  }
  return response;
}

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  std::string server_address = absl::GetFlag(FLAGS_server_address);
  std::string model_name = absl::GetFlag(FLAGS_model_name);
  int32_t batch_size = absl::GetFlag(FLAGS_batch_size);
  int32_t top_k = absl::GetFlag(FLAGS_top_k);
  std::string labels_file = absl::GetFlag(FLAGS_labels_file);
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

  std::string image_file = absl::GetFlag(FLAGS_image_file);
  int32_t num_requests = absl::GetFlag(FLAGS_num_requests);
  int32_t warmup_requests = absl::GetFlag(FLAGS_warmup_requests);
  if (batch_size < 1) {
    LOG(ERROR) << "--batch_size must be >= 1, got " << batch_size;
    return 1;
  }
  if (num_requests < 1) {
    LOG(ERROR) << "--num_requests must be >= 1, got " << num_requests;
    return 1;
  }
  if (warmup_requests < 0) {
    LOG(ERROR) << "--warmup_requests must be >= 0, got " << warmup_requests;
    return 1;
  }
  std::vector<std::string> labels = LoadLabels(labels_file);
  bool verbose = absl::GetFlag(FLAGS_verbose);

  // Build the gRPC stub once so per-request latencies don't include channel
  // setup cost.
  auto creds_or = xla_serving::serving::CreateChannelCredentials(
      server_cert_chain_path, use_insecure_grpc_credentials);
  if (!creds_or.ok()) {
    LOG(ERROR) << "Failed to create channel credentials: " << creds_or.status();
    return 1;
  }
  grpc::ChannelArguments channel_args;
  channel_args.SetMaxSendMessageSize(-1);
  channel_args.SetMaxReceiveMessageSize(-1);
  std::shared_ptr<grpc::Channel> channel =
      grpc::CreateCustomChannel(server_address, *creds_or, channel_args);
  std::unique_ptr<xla_serving::serving::PredictionService::Stub> stub =
      xla_serving::serving::PredictionService::NewStub(channel);
  LOG(INFO) << "Connected to " << server_address;

  // Warmup phase (discard these timings)
  for (int i = 0; i < warmup_requests; ++i) {
    auto result = RunPredict(stub.get(), model_name, batch_size, image_file,
                             verbose);
    if (!result.ok()) {
      LOG(ERROR) << "Warmup request failed: " << result.status();
      return 1;
    }
    LOG(INFO) << "Warmup request " << (i + 1) << "/" << warmup_requests
              << " done.";
  }

  // Benchmark phase
  std::vector<double> latencies_ms;
  latencies_ms.reserve(num_requests);
  xla_serving::serving::PredictResponse last_response;

  for (int i = 0; i < num_requests; ++i) {
    absl::Time start = absl::Now();
    auto predict_response = RunPredict(stub.get(), model_name, batch_size,
                                       image_file, verbose);
    double elapsed_ms = absl::ToDoubleMilliseconds(absl::Now() - start);

    if (!predict_response.ok()) {
      LOG(ERROR) << "Request " << (i + 1)
                 << " failed: " << predict_response.status();
      return 1;
    }
    latencies_ms.push_back(elapsed_ms);

    if (i == num_requests - 1) {
      last_response = *std::move(predict_response);
    }
  }

  LogPredictResponse(last_response, top_k, labels);

  if (num_requests > 1) {
    // Compute statistics. Percentiles use the nearest-rank method (rounded)
    // and clamp to size-1 so pXX never reads past the end for small samples.
    std::sort(latencies_ms.begin(), latencies_ms.end());
    double sum = 0;
    for (double d : latencies_ms) sum += d;
    double avg = sum / latencies_ms.size();
    double min = latencies_ms.front();
    double max = latencies_ms.back();
    auto percentile = [&](double p) {
      size_t idx = static_cast<size_t>(std::round(p * latencies_ms.size()));
      if (idx >= latencies_ms.size()) idx = latencies_ms.size() - 1;
      return latencies_ms[idx];
    };
    double p50 = percentile(0.50);
    double p90 = percentile(0.90);
    double p99 = percentile(0.99);

    LOG(INFO) << "========== Benchmark Results ==========";
    LOG(INFO) << "Requests: " << num_requests
              << " (after " << warmup_requests << " warmup)";
    LOG(INFO) << "Latency (ms):";
    LOG(INFO) << "  Min:  " << min;
    LOG(INFO) << "  Avg:  " << avg;
    LOG(INFO) << "  P50:  " << p50;
    LOG(INFO) << "  P90:  " << p90;
    LOG(INFO) << "  P99:  " << p99;
    LOG(INFO) << "  Max:  " << max;
    LOG(INFO) << "Throughput: " << (1000.0 / avg) << " req/s (sequential)";
  }

  return 0;
}
