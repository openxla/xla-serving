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

#include "runtime/serving/gcp/server/orbax_grpc_server.h"  // from @neptune

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/flags/flag.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/time/time.h"  // from @abseil-cpp
#include "include/grpcpp/server_context.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/support/status.h"  // from @com_github_grpc_grpc
#include "runtime/core/host_buffer.h"  // from @neptune
#include "runtime/core/proto/host_buffer.pb.h"  // from @neptune
#include "runtime/core/value.h"  // from @neptune
#include "runtime/interpreter/interpreter.h"  // from @neptune
#include "runtime/interpreter/run_options.h"  // from @neptune
#include "runtime/kernel/kernel_io_data_type.h"  // from @neptune
#include "runtime/serving/gcp/proto/model.pb.h"  // from @neptune
#include "runtime/serving/gcp/proto/prediction_service.pb.h"  // from @neptune
#if ENABLE_TF == 1
  include "third_party/neptune/runtime/tensorflow/tf_compatible_interpreter_create_options.h"
#else
  #include "runtime/interpreter/interpreter_create_options.h"  // from @neptune  
#endif
#include "runtime/tensorflow/utils/tf_to_host_buffer.h"  // from @neptune
#include "runtime/serving/gcp/server/grpc_status_utils.h"  // from @neptune
#include "xla/pjrt/pjrt_api.h"  // from @xla
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"  // from @xla
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"  // from @xla
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"  // from @xla
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.h"  // from @xla
#include "xla/pjrt/plugin/xla_tpu/xla_tpu_pjrt_client.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_client.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "tensorflow/core/framework/tensor.h"  // from @org_tensorflow
#include "tsl/profiler/lib/traceme.h"  // from @tsl
#include "tsl/profiler/lib/traceme_encode.h"  // from @tsl
#include "runtime/serving/gcp/server/flags.h"  // from @neptune

namespace neptune {
namespace serving {

namespace {

template <typename T>
void LogMapKeys(const absl::flat_hash_map<std::string, T>& map,
                absl::string_view map_name, int vlog_level) {
  if (VLOG_IS_ON(vlog_level)) {
    std::vector<std::string> keys;
    for (const auto& pair : map) {
      keys.push_back(pair.first);
    }
    VLOG(vlog_level) << map_name << ": [" << absl::StrJoin(keys, ", ") << "]";
  }
}

}  // namespace

absl::StatusOr<std::unique_ptr<OrbaxGrpcServer>> OrbaxGrpcServer::Create(
    absl::string_view model_name, absl::string_view model_path) {
  // The `orbax_platform` string refers to the Orbax `Platform` enum.
  std::string orbax_platform = absl::GetFlag(FLAGS_orbax_grpc_server_platform);
  std::string pjrt_platform = orbax_platform;
  // Orbax models and the manifest use "cuda", but the underlying PJRT API
  // explicitly requires the plugin string "gpu" for initialization
  if (orbax_platform == "cuda") {
    pjrt_platform = "gpu";
  }

  LOG(INFO) << "Checking PJRT plugin initialization for PJRT platform: '"
            << pjrt_platform << "'";
  TF_ASSIGN_OR_RETURN(bool is_pjrt_plugin_initialized,
                      pjrt::IsPjrtPluginInitialized(pjrt_platform));
  if (!is_pjrt_plugin_initialized) {
    LOG(INFO) << absl::StrCat("PJRT plugin not initialized for PJRT platform '",
                              pjrt_platform, "'. Initializing.");
    TF_RETURN_IF_ERROR(pjrt::InitializePjrtPlugin(pjrt_platform));
  }
  LOG(INFO) << "PJRT plugin initialized for PJRT platform: '" << pjrt_platform
            << "'. Getting client.";
  std::shared_ptr<xla::ifrt::Client> client;
  if (pjrt_platform == "cpu") {
    TF_ASSIGN_OR_RETURN(auto pjrt_client,
                        xla::GetXlaPjrtCpuClient(xla::CpuClientOptions()));
    client = xla::ifrt::PjRtClient::Create(std::move(pjrt_client));
  } else if (pjrt_platform == "tpu") {
    TF_ASSIGN_OR_RETURN(auto pjrt_client, xla::GetXlaPjrtTpuClient());
    client = xla::ifrt::PjRtClient::Create(std::move(pjrt_client));
  } else if (pjrt_platform == "gpu") {
    TF_ASSIGN_OR_RETURN(auto pjrt_client,
                        xla::GetXlaPjrtGpuClient(xla::GpuClientOptions()));
    client = xla::ifrt::PjRtClient::Create(std::move(pjrt_client));
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported PJRT platform: ", pjrt_platform));
  }
  LOG(INFO) << "Orbax Server initialized for Orbax platform: " << orbax_platform
            << " (Device count: " << client->device_count() << ")";

  auto execute_worker_thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "OrbaxExecutionThreadPool",
      kThreadPoolNumThreads);
  auto computation_thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "OrbaxComputationThreadPool",
      kThreadPoolNumThreads);
  auto checkpoint_loader_thread_pool =
      std::make_unique<tsl::thread::ThreadPool>(
          tsl::Env::Default(), tsl::ThreadOptions(),
          "OrbaxCheckpointLoaderThreadPool", kThreadPoolNumThreads);
  // Support model version in ModelRunner if we see use cases.
  #if ENABLE_TF == 1
    auto create_options =
        std::make_unique<neptune::TfCompatibleInterpreterCreateOptions>();
  #else
    auto create_options = std::make_unique<neptune::InterpreterCreateOptions>();
  #endif

  create_options->checkpoint_loader_thread_pool =
      checkpoint_loader_thread_pool.get();
  create_options->model_name = model_name;
  create_options->model_version = 0;
  create_options->model_path = model_path;
  create_options->execute_worker_thread_pool = execute_worker_thread_pool.get();
  create_options->ifrt_client = client;
  create_options->computation_thread_pool = computation_thread_pool.get();
  create_options->platform_name = orbax_platform;
  std::unique_ptr<neptune::Interpreter> interpreter;
  TF_ASSIGN_OR_RETURN(interpreter,
                      neptune::Interpreter::Create(std::move(create_options)));
  return std::make_unique<OrbaxGrpcServer>(
      model_name, std::move(interpreter), client,
      std::move(execute_worker_thread_pool), std::move(computation_thread_pool),
      std::move(checkpoint_loader_thread_pool));
}

grpc::Status OrbaxGrpcServer::Predict(
    grpc::ServerContext* context, const neptune::serving::PredictRequest* request,
    neptune::serving::PredictResponse* response) {
  VLOG(4) << "Received PredictRequest: " << request->DebugString();
  if (interpreter_ == nullptr) {
    return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                        "Interpreter not initialized.");
  }

  absl::string_view pipeline_name =
      request->model_spec().signature_name().empty()
          ? absl::string_view(kDefaultServingSignatureDefKey)
          : absl::string_view(request->model_spec().signature_name());
  VLOG(1) << "Using pipeline name: '" << pipeline_name << "'";

  tsl::profiler::TraceMe trace_predict([&]() {
    return tsl::profiler::TraceMeEncode("Predict",
                                        {{"pipeline_name", pipeline_name}});
  });

  auto input_data_type_or = interpreter_->GetModelInputDataType(pipeline_name);
  if (!input_data_type_or.ok()) {
    return ToGrpcStatus(input_data_type_or.status());
  }
  neptune::KernelIODataType input_data_type = input_data_type_or.value();
  bool input_expects_tensor =
      (input_data_type == neptune::KernelIODataType::kTensor);
  VLOG(3) << "Input expects tensor: " << input_expects_tensor;

  absl::flat_hash_map<std::string, neptune::Value> input_with_names;
  {
    tsl::profiler::TraceMe trace_inputs("ConvertInputs");
    for (const auto& [name, host_buffer_proto] : request->inputs()) {
      auto host_buffer_or = neptune::HostBuffer::FromProto(host_buffer_proto);
      if (!host_buffer_or.ok()) {
        return ToGrpcStatus(host_buffer_or.status());
      }
      neptune::HostBuffer host_buffer = std::move(host_buffer_or).value();
      if (input_expects_tensor) {
        auto tensor_or = ToTFTensor(host_buffer);
        if (!tensor_or.ok()) {
          return ToGrpcStatus(tensor_or.status());
        }
        input_with_names[name] = neptune::Value(std::move(tensor_or).value());
      } else {
        input_with_names[name] = neptune::Value(std::move(host_buffer));
      }
    }
  }
  LogMapKeys(input_with_names, "Input with names", /*vlog_level=*/2);

  neptune::RunOptions run_options;
  run_options.deadline = absl::FromChrono(context->deadline());
  auto output_with_names_or = interpreter_->RunWithMapReturn(pipeline_name, /*positional_inputs=*/{},
                                     input_with_names, run_options);
  if (!output_with_names_or.ok()) {
    return ToGrpcStatus(output_with_names_or.status());
  }
  auto output_with_names = std::move(output_with_names_or).value();
  LogMapKeys(output_with_names, "Output with names", /*vlog_level=*/2);

  auto output_data_type_or = interpreter_->GetModelOutputDataType(pipeline_name);
  if (!output_data_type_or.ok()) {
    return ToGrpcStatus(output_data_type_or.status());
  }
  neptune::KernelIODataType output_data_type = output_data_type_or.value();
  bool output_is_tensor = (output_data_type == neptune::KernelIODataType::kTensor);
  VLOG(3) << "Output is tensor: " << output_is_tensor;

  {
    tsl::profiler::TraceMe traceme_serialization("ConvertOutputs");
    for (const auto& [name, value] : output_with_names) {
      neptune::HostBuffer host_buffer;
      if (output_is_tensor) {
        auto host_buffer_or = ToHostBuffer(value.Get<tensorflow::Tensor>());
        if (!host_buffer_or.ok()) {
          return ToGrpcStatus(host_buffer_or.status());
        }
        host_buffer = std::move(host_buffer_or).value();
      } else {
        host_buffer = value.Get<neptune::HostBuffer>();
      }
      HostBufferProto host_buffer_proto =
          HostBuffer::ToHostBufferProto(std::move(host_buffer));
      (*response->mutable_outputs())[name] = std::move(host_buffer_proto);
    }
  }

  response->mutable_model_spec()->set_name(model_name_);
  response->mutable_model_spec()->set_signature_name(pipeline_name);

  return grpc::Status::OK;
}

grpc::Status OrbaxGrpcServer::GetModelMetadata(
    grpc::ServerContext* context,
    const neptune::serving::GetModelMetadataRequest* request,
    neptune::serving::GetModelMetadataResponse* response) {
  return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, "Not implemented");
}

}  // namespace serving
}  // namespace neptune
