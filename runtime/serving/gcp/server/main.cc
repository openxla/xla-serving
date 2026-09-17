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
#include <cstdint>
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
#include "runtime/serving/gcp/auth/utils.h"  // from @neptune
#include "runtime/serving/gcp/server/orbax_callback_grpc_server.h"  // from @neptune
#include "xla/tsl/platform/status.h"  // from @xla
#include "xla/tsl/profiler/rpc/profiler_server.h"  // from @xla
#include "tsl/platform/init_main.h"  // from @tsl

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

ABSL_FLAG(bool, enable_lazy_loading, true, "Enable lazy loading");
ABSL_FLAG(bool, lazy_loading_use_graph_executor, true,
          "Lazy loading use graph executor");
ABSL_FLAG(bool, enable_mlrt, true, "Enable MLRT");
ABSL_FLAG(bool, allow_xla_cpu, true, "Allow XLA CPU");
ABSL_FLAG(bool, hoist_invariant_ops, true, "Hoist invariant ops");
ABSL_FLAG(bool, sink_in_invariant_ops, true, "Sink in invariant ops");
ABSL_FLAG(int64_t, cost_threshold, 10000, "Cost threshold");
ABSL_FLAG(int, xprof_port, 8888, "Port for the XProf server to listen on.");
ABSL_FLAG(bool, warmup_server, false,
          "Warmup server on startup. This requires the warmup requests file "
          "('assets.extra/neptune_serving_warmup_requests') to be present in "
          "the model directory. The file must contain size-prefixed records "
          "(varint32 length prefix followed by serialized proto bytes) of "
          "neptune::serving::PredictionLog containing PredictRequests. This file "
          "can be generated automatically during model export by passing "
          "`warmup_examples` to `neptune_model.save()`. See "
          "https://github.com/AI-Hypercomputer/neptune/blob/main/"
          "README.md#server-warmup-on-startup for details.");
ABSL_FLAG(int, max_receive_message_size,
          16 * 1024 * 1024,  // 16 MiB, up from default 4 MiB
          "Maximum size of a received message.");
ABSL_FLAG(int, max_send_message_size, -1,  // Unlimited, matches gRPC default
          "Maximum size of a send message.");
ABSL_FLAG(int, thread_pool_num_threads, 0,
          "Number of threads for server execution thread pool. If 0, defaults "
          "to 4.");
ABSL_FLAG(int, response_threads, 0,
          "Threads for building Predict responses and finishing their RPCs. "
          "0 = host parallelism; -1 = no pool, finish inline on the completing "
          "thread.");

void RunServer() {
  std::string server_address = absl::StrCat("[::]:", absl::GetFlag(FLAGS_port));

  neptune::serving::OrbaxCallbackGrpcServerOptions options;
  options.enable_lazy_loading = absl::GetFlag(FLAGS_enable_lazy_loading);
  options.lazy_loading_use_graph_executor =
      absl::GetFlag(FLAGS_lazy_loading_use_graph_executor);
  options.enable_mlrt = absl::GetFlag(FLAGS_enable_mlrt);
  options.allow_xla_cpu = absl::GetFlag(FLAGS_allow_xla_cpu);
  options.hoist_invariant_ops = absl::GetFlag(FLAGS_hoist_invariant_ops);
  options.sink_in_invariant_ops = absl::GetFlag(FLAGS_sink_in_invariant_ops);
  options.cost_threshold = absl::GetFlag(FLAGS_cost_threshold);
  options.warmup_server = absl::GetFlag(FLAGS_warmup_server);
  options.thread_pool_num_threads =
      absl::GetFlag(FLAGS_thread_pool_num_threads);
  options.response_threads = absl::GetFlag(FLAGS_response_threads);

  LOG(INFO) << "Server address: " << server_address;
  auto service_or = neptune::serving::OrbaxCallbackGrpcServer::Create(
      absl::GetFlag(FLAGS_model_name), absl::GetFlag(FLAGS_model_path),
      options);
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

  builder.SetMaxReceiveMessageSize(
      absl::GetFlag(FLAGS_max_receive_message_size));
  builder.SetMaxSendMessageSize(absl::GetFlag(FLAGS_max_send_message_size));
  builder.AddListeningPort(server_address, *creds);
  builder.RegisterService(service_or.value().get());
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  if (server == nullptr) {
    LOG(FATAL) << "Failed to build server and start server";
  }
  LOG(INFO) << "JAX Serving listening on " << server_address;

  server->Wait();
}

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  tsl::port::InitMain(argv[0], &argc, &argv);
  LOG(INFO) << "Starting Neptune Server on GCP with flags: "
            << absl::GetFlag(FLAGS_model_name) << " "
            << absl::GetFlag(FLAGS_model_path) << " "
            << absl::GetFlag(FLAGS_server_private_key_path) << " "
            << absl::GetFlag(FLAGS_server_cert_chain_path) << " "
            << absl::GetFlag(FLAGS_client_cert_chain_path) << " "
            << absl::GetFlag(FLAGS_use_insecure_grpc_credentials);
  tsl::profiler::ProfilerServer server;
  server.StartProfilerServer(absl::GetFlag(FLAGS_xprof_port));
  RunServer();
  return 0;
}
