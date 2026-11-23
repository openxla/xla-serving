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
#include <cstddef>
#if ENABLE_TF == 1
#include <cstdint>
#endif
#include <memory>
#include <string>

#include "absl/flags/flag.h"  // from @abseil-cpp
#include "absl/flags/parse.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "include/grpcpp/security/server_credentials.h"  // from @grpc
#include "include/grpcpp/server.h"  // from @grpc
#include "include/grpcpp/server_builder.h"  // from @grpc
#include "protos/grpc/model_server_config.pb.h"  // from @xla_serving
#include "runtime/serving/gcp/auth/utils.h"  // from @xla_serving
#include "runtime/serving/gcp/server/orbax_callback_grpc_server.h"  // from @xla_serving
#include "google/protobuf/text_format.h"
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/status.h"  // from @xla
// clang-format off
#include "xla/tsl/profiler/rpc/profiler_server.h"  // from @xla
// clang-format on
#include "tsl/platform/init_main.h"  // from @tsl

ABSL_FLAG(int, port, 8500, "Port for the gRPC server to listen on.");

ABSL_FLAG(std::string, model_name, "", "Name of the model.");
ABSL_FLAG(std::string, model_path, "", "Path to the model.");
ABSL_FLAG(std::string, model_config_file, "",
          "Path to ModelServerConfig file in text format.");

ABSL_FLAG(std::string, server_private_key_path, "",
          "Path to the server's private key file for SSL/TLS.");
ABSL_FLAG(std::string, server_cert_chain_path, "",
          "Path to the server's PEM-encoded certificate chain file for "
          "SSL/TLS.");
ABSL_FLAG(
    std::string, client_cert_chain_path, "",
    "Optional path to the certificate chain file for client authentication.");
ABSL_FLAG(bool, use_alts_credentials, false,
          "Whether to use Application Layer Transport Security (ALTS) "
          "credentials. Supported on Google Cloud (Compute Engine, GKE, Cloud "
          "TPU) with automatic mutual TLS and no certificate files.");
ABSL_FLAG(bool, use_insecure_grpc_credentials, false,
          "Whether to use insecure credentials for the server. WARNING: THIS "
          "SHOULD NOT BE USED IN PRODUCTION.");

#if ENABLE_TF == 1
ABSL_FLAG(bool, enable_lazy_loading, true, "Enable lazy loading");
ABSL_FLAG(bool, lazy_loading_use_graph_executor, true,
          "Lazy loading use graph executor");
ABSL_FLAG(bool, enable_mlrt, true, "Enable MLRT");
ABSL_FLAG(bool, allow_xla_cpu, true, "Allow XLA CPU");
ABSL_FLAG(bool, hoist_invariant_ops, true, "Hoist invariant ops");
ABSL_FLAG(bool, sink_in_invariant_ops, true, "Sink in invariant ops");
ABSL_FLAG(int64_t, cost_threshold, 10000, "Cost threshold");
#endif
ABSL_FLAG(int, xprof_port, 8888, "Port for the XProf server to listen on.");
ABSL_FLAG(bool, warmup_server, false,
          "Warmup server on startup. This requires the warmup requests file "
          "('assets.extra/xla_serving_warmup_requests') to be present in "
          "the model directory. The file must contain size-prefixed records "
          "(varint32 length prefix followed by serialized proto bytes) of "
          "xla_serving::serving::PredictionLog containing PredictRequests. This file "
          "can be generated automatically during model export by passing "
          "`warmup_examples` to `xla_saved_model.save()`. See "
          "https://github.com/openxla/xla-serving/blob/main/"
          "README.md#server-warmup-on-startup for details.");
ABSL_FLAG(bool, warmup_all_batch_sizes, false,
          "If true, warms up all allowed batch sizes during server warmup.");
ABSL_FLAG(int, max_receive_message_size,
          16 * 1024 * 1024,  // 16 MiB, up from default 4 MiB
          "Maximum size of a received message.");
ABSL_FLAG(int, max_send_message_size, -1,  // Unlimited, matches gRPC default
          "Maximum size of a send message.");
ABSL_FLAG(int, thread_pool_num_threads, 0,
          "Number of threads for server execution thread pool. If 0, defaults "
          "to 4.");
ABSL_FLAG(bool, use_nanort_cpu_client, true,
          "CPU only. If true, uses the NanoRt IFRT client instead of the PjRt "
          "CPU client.");
ABSL_FLAG(bool, nanort_share_computation_thread_pool, false,
          "Only used when --use_nanort_cpu_client is true. If true, the NanoRt "
          "client uses the server's computation thread pool as its intra-op "
          "thread pool instead of creating its own.");
ABSL_FLAG(bool, tpu_use_premapped_buffer, false,
          "TPU only. If true, enables premapped buffer in the PjRt TPU "
          "client.");
ABSL_FLAG(size_t, tpu_premapped_buffer_size, 2 * (1ULL << 30),
          "TPU only. Size in bytes of the premapped buffer for the PjRt TPU "
          "client when --tpu_use_premapped_buffer is true.");
ABSL_FLAG(int, tpu_num_premapped_partitions_per_device, 1,
          "TPU only. Number of premapped partitions per device for the PjRt "
          "TPU client when --tpu_use_premapped_buffer is true.");

void RunServer() {
  std::string private_key_path = absl::GetFlag(FLAGS_server_private_key_path);
  std::string cert_chain_path = absl::GetFlag(FLAGS_server_cert_chain_path);
  std::string root_certs_path = absl::GetFlag(FLAGS_client_cert_chain_path);
  bool use_alts_credentials = absl::GetFlag(FLAGS_use_alts_credentials);
  bool use_insecure_grpc_credentials =
      absl::GetFlag(FLAGS_use_insecure_grpc_credentials);

  const bool use_local_credentials = xla_serving::serving::UsesLocalServerCredentials(
      private_key_path, cert_chain_path, root_certs_path,
      use_insecure_grpc_credentials, use_alts_credentials);
  std::string server_address =
      absl::StrCat(use_local_credentials ? "localhost:" : "[::]:",
                   absl::GetFlag(FLAGS_port));

  std::string model_config_file = absl::GetFlag(FLAGS_model_config_file);
  std::string model_name = absl::GetFlag(FLAGS_model_name);
  std::string model_path = absl::GetFlag(FLAGS_model_path);
  bool has_config = !model_config_file.empty();
  bool has_model_name = !model_name.empty();
  bool has_model_path = !model_path.empty();

  if (has_config && (has_model_name || has_model_path)) {
    LOG(FATAL) << "Cannot specify both --model_config_file and "
                  "--model_name/--model_path.";
  }
  if (!has_config && (!has_model_name || !has_model_path)) {
    LOG(FATAL) << "Either --model_config_file or both --model_name and "
                  "--model_path must be specified.";
  }

  absl::StatusOr<std::shared_ptr<grpc::ServerCredentials>> creds =
      xla_serving::serving::CreateServerCredentials(
          private_key_path, cert_chain_path, root_certs_path,
          use_insecure_grpc_credentials, use_alts_credentials);
  TF_CHECK_OK(creds.status());

  xla_serving::serving::OrbaxCallbackGrpcServerOptions options;
#if ENABLE_TF == 1
  options.enable_lazy_loading = absl::GetFlag(FLAGS_enable_lazy_loading);
  options.lazy_loading_use_graph_executor =
      absl::GetFlag(FLAGS_lazy_loading_use_graph_executor);
  options.enable_mlrt = absl::GetFlag(FLAGS_enable_mlrt);
  options.allow_xla_cpu = absl::GetFlag(FLAGS_allow_xla_cpu);
  options.hoist_invariant_ops = absl::GetFlag(FLAGS_hoist_invariant_ops);
  options.sink_in_invariant_ops = absl::GetFlag(FLAGS_sink_in_invariant_ops);
  options.cost_threshold = absl::GetFlag(FLAGS_cost_threshold);
#endif
  options.warmup_server = absl::GetFlag(FLAGS_warmup_server);
  options.warmup_all_batch_sizes = absl::GetFlag(FLAGS_warmup_all_batch_sizes);
  options.thread_pool_num_threads =
      absl::GetFlag(FLAGS_thread_pool_num_threads);
  options.use_nanort_cpu_client = absl::GetFlag(FLAGS_use_nanort_cpu_client);
  options.nanort_share_computation_thread_pool =
      absl::GetFlag(FLAGS_nanort_share_computation_thread_pool);
  options.tpu_use_premapped_buffer =
      absl::GetFlag(FLAGS_tpu_use_premapped_buffer);
  options.tpu_premapped_buffer_size =
      absl::GetFlag(FLAGS_tpu_premapped_buffer_size);
  options.tpu_num_premapped_partitions_per_device =
      absl::GetFlag(FLAGS_tpu_num_premapped_partitions_per_device);

  LOG(INFO) << "Server address: " << server_address;
  absl::StatusOr<std::unique_ptr<xla_serving::serving::OrbaxCallbackGrpcServer>>
      service_or;
  if (has_config) {
    std::string config_content;
    TF_CHECK_OK(tsl::ReadFileToString(tsl::Env::Default(), model_config_file,
                                      &config_content));
    xla_serving::serving::ModelServerConfig model_server_config;
    if (!google::protobuf::TextFormat::ParseFromString(config_content,
                                             &model_server_config)) {
      LOG(FATAL) << "Failed to parse ModelServerConfig from "
                 << model_config_file;
    }
    service_or = xla_serving::serving::OrbaxCallbackGrpcServer::Create(
        model_server_config, options);
  } else {
    service_or = xla_serving::serving::OrbaxCallbackGrpcServer::Create(
        model_name, model_path, options);
  }
  TF_CHECK_OK(service_or.status());

  grpc::ServerBuilder builder;
  int selected_port = 0;
  builder.SetMaxReceiveMessageSize(
      absl::GetFlag(FLAGS_max_receive_message_size));
  builder.SetMaxSendMessageSize(absl::GetFlag(FLAGS_max_send_message_size));
  builder.AddListeningPort(server_address, *creds, &selected_port);
  builder.RegisterService(service_or.value().get());
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  if (server == nullptr || selected_port == 0) {
    LOG(FATAL) << "Failed to build and start server on " << server_address;
  }
  LOG(INFO) << "JAX Serving listening on "
            << (use_local_credentials ? "localhost:" : "[::]:")
            << selected_port;

  server->Wait();
}

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  tsl::port::InitMain(argv[0], &argc, &argv);
  LOG(INFO) << "Starting XSE Server on GCP with flags: "
            << absl::GetFlag(FLAGS_model_name) << " "
            << absl::GetFlag(FLAGS_model_path) << " "
            << absl::GetFlag(FLAGS_model_config_file) << " "
            << absl::GetFlag(FLAGS_server_private_key_path) << " "
            << absl::GetFlag(FLAGS_server_cert_chain_path) << " "
            << absl::GetFlag(FLAGS_client_cert_chain_path) << " "
            << absl::GetFlag(FLAGS_use_alts_credentials) << " "
            << absl::GetFlag(FLAGS_use_insecure_grpc_credentials);
  tsl::profiler::ProfilerServer server;
  server.StartProfilerServer(absl::GetFlag(FLAGS_xprof_port));
  RunServer();
  return 0;
}
