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

#include "runtime/serving/gcp/server/orbax_callback_grpc_server.h"  // from @neptune

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/flags/flag.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/time/time.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "include/grpcpp/server_context.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/support/server_callback.h"  // from @com_github_grpc_grpc
#include "include/grpcpp/support/status.h"  // from @com_github_grpc_grpc
#include "protos/grpc/host_buffer.pb.h"  // from @neptune
#include "protos/grpc/model.pb.h"  // from @neptune
#include "protos/grpc/prediction_log.pb.h"  // from @neptune
#include "protos/grpc/prediction_service.pb.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune
#include "runtime/core/value.h"  // from @neptune
#include "runtime/interpreter/interpreter.h"  // from @neptune
#include "runtime/interpreter/run_options.h"  // from @neptune
#include "runtime/kernel/kernel_io_data_type.h"  // from @neptune
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/util/delimited_message_util.h"
#include "xla/tsl/platform/file_system.h"  // from @xla
#if ENABLE_TF == 1
  #include "runtime/tensorflow/tf_compatible_interpreter_create_options.h"  // from @neptune
#else
  #include "runtime/interpreter/interpreter_create_options.h"  // from @neptune
#endif
#include "runtime/tensorflow/utils/tf_to_host_buffer.h"  // from @neptune
#include "runtime/serving/gcp/server/grpc_status_utils.h"  // from @neptune
#include "xla/pjrt/pjrt_api.h"  // from @xla
#include "runtime/serving/gcp/server/ifrt_client_creator.h"  // from @neptune
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_client.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "tensorflow/core/framework/tensor.h"  // from @org_tensorflow
#include "runtime/serving/gcp/server/flags.h"  // from @neptune
#include "tsl/profiler/lib/traceme.h"  // from @tsl

namespace neptune {
namespace serving {

namespace {

static constexpr char kDefaultServingSignatureDefKey[] = "serving_default";
static constexpr char kWarmupRequestsRelativePath[] =
    "assets.extra/neptune_serving_warmup_requests";

#if ENABLE_TF == 1
tensorflow::tfrt_stub::SavedModel::Options GetSavedModelOptions(
    const OrbaxCallbackGrpcServerOptions& server_options) {
  static tensorflow::tfrt_stub::Runtime* default_runtime =
      tensorflow::tfrt_stub::Runtime::Create(
          tensorflow::tfrt_stub::WrapDefaultWorkQueue(
              tfrt::CreateMultiThreadedWorkQueue(8, 8)))
          .release();
  CHECK(default_runtime);
  tensorflow::tfrt_stub::SavedModel::Options options(default_runtime);
  options.enable_lazy_loading = server_options.enable_lazy_loading;
  options.lazy_loading_use_graph_executor =
      server_options.lazy_loading_use_graph_executor;
  options.graph_execution_options.enable_mlrt = server_options.enable_mlrt;
  options.graph_execution_options.compile_options.allow_xla_cpu =
      server_options.allow_xla_cpu;
  options.graph_execution_options.compile_options.hoist_invariant_ops =
      server_options.hoist_invariant_ops;
  options.graph_execution_options.compile_options.sink_in_invariant_ops =
      server_options.sink_in_invariant_ops;
  options.graph_execution_options.compile_options.cost_threshold =
      server_options.cost_threshold;
  return options;
}

#endif



}  // namespace

absl::StatusOr<std::unique_ptr<OrbaxCallbackGrpcServer>>
OrbaxCallbackGrpcServer::Create(absl::string_view model_name,
                                absl::string_view model_path,
                                const OrbaxCallbackGrpcServerOptions& options) {
  std::string orbax_platform = absl::GetFlag(FLAGS_orbax_grpc_server_platform);
  LOG(INFO) << "Creating Orbax Server for Orbax platform: " << orbax_platform;
  std::string pjrt_platform = orbax_platform;
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
  TF_ASSIGN_OR_RETURN(client, GetIfrtClient(pjrt_platform));
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
  #if ENABLE_TF == 1
    auto create_options =
        std::make_unique<neptune::TfCompatibleInterpreterCreateOptions>();
    create_options->tfrt_options.saved_model_options =
        GetSavedModelOptions(options);
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
    auto server = std::make_unique<OrbaxCallbackGrpcServer>(
        model_name, std::move(interpreter), client,
        std::move(execute_worker_thread_pool),
        std::move(computation_thread_pool),
        std::move(checkpoint_loader_thread_pool));
    if (options.warmup_server) {
      TF_RETURN_IF_ERROR(server->Warmup(model_path));
    }
    return server;
}

grpc::ServerUnaryReactor* OrbaxCallbackGrpcServer::Predict(
    grpc::CallbackServerContext* context,
    const neptune::serving::PredictRequest* request,
    neptune::serving::PredictResponse* response) {
  VLOG(4) << "Received PredictRequest: " << request->DebugString();
  grpc::ServerUnaryReactor* reactor = context->DefaultReactor();

  if (interpreter_ == nullptr) {
    reactor->Finish(grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                                 "Interpreter not initialized."));
    return reactor;
  }

  absl::string_view pipeline_name =
      request->model_spec().signature_name().empty()
          ? absl::string_view(kDefaultServingSignatureDefKey)
          : absl::string_view(request->model_spec().signature_name());
  VLOG(1) << "Using pipeline name: '" << pipeline_name << "'";

  neptune::RunOptions run_options;
  run_options.deadline = absl::FromChrono(context->deadline());

  absl::StatusOr<absl::flat_hash_map<std::string, neptune::Value>> output_with_names =
      RunPredictInternal(*request, pipeline_name, run_options);
  if (!output_with_names.ok()) {
    reactor->Finish(ToGrpcStatus(output_with_names.status()));
    return reactor;
  }
  absl::StatusOr<neptune::KernelIODataType> output_data_type =
      interpreter_->GetModelOutputDataType(pipeline_name);
  if (!output_data_type.ok()) {
    reactor->Finish(ToGrpcStatus(output_data_type.status()));
    return reactor;
  }
  bool output_is_tensor = (*output_data_type == neptune::KernelIODataType::kTensor);

  {
    tsl::profiler::TraceMe traceme_serialization("ConvertOutputs");
    for (const auto& [name, value] : *output_with_names) {
      absl::StatusOr<neptune::HostBuffer> host_buffer;
      if (output_is_tensor) {
         host_buffer = ToHostBuffer(value.Get<tensorflow::Tensor>());
      } else {
        host_buffer = value.Get<neptune::HostBuffer>();
      }
      if (!host_buffer.ok()) {
        reactor->Finish(ToGrpcStatus(host_buffer.status()));
        return reactor;
      }
      HostBufferProto host_buffer_proto =
          HostBuffer::ToHostBufferProto(std::move(*host_buffer));
      (*response->mutable_outputs())[name] = std::move(host_buffer_proto);
    }
  }

  response->mutable_model_spec()->set_name(model_name_);
  response->mutable_model_spec()->set_signature_name(pipeline_name);
  reactor->Finish(grpc::Status::OK);
  return reactor;
}

absl::StatusOr<absl::flat_hash_map<std::string, neptune::Value>>
OrbaxCallbackGrpcServer::RunPredictInternal(
    const neptune::serving::PredictRequest& request,
    absl::string_view pipeline_name, const neptune::RunOptions& run_options) {
  TF_ASSIGN_OR_RETURN(neptune::KernelIODataType input_data_type,
                      interpreter_->GetModelInputDataType(pipeline_name));
  bool input_expects_tensor =
      (input_data_type == neptune::KernelIODataType::kTensor);

  absl::flat_hash_map<std::string, neptune::Value> input_with_names;
  input_with_names.reserve(request.inputs().size());
  for (const auto& [name, host_buffer_proto] : request.inputs()) {
    TF_ASSIGN_OR_RETURN(neptune::HostBuffer host_buffer,
                        neptune::HostBuffer::FromProto(host_buffer_proto));
    if (input_expects_tensor) {
      TF_ASSIGN_OR_RETURN(tensorflow::Tensor tensor, ToTFTensor(host_buffer));
      input_with_names[name] = neptune::Value(std::move(tensor));
    } else {
      input_with_names[name] = neptune::Value(std::move(host_buffer));
    }
  }

  return interpreter_->RunWithMapReturn(
      pipeline_name, /*positional_inputs=*/{}, input_with_names,
      run_options);
}

absl::Status OrbaxCallbackGrpcServer::Warmup(absl::string_view model_path) {
  std::string warmup_path =
      absl::StrCat(model_path, "/", kWarmupRequestsRelativePath);
  tsl::Env* env = tsl::Env::Default();
  TF_RETURN_IF_ERROR(env->FileExists(warmup_path));

  LOG(INFO) << "Running warmup from " << warmup_path;

  std::unique_ptr<tsl::RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(warmup_path, &file));
  uint64_t file_size;
  TF_RETURN_IF_ERROR(env->GetFileSize(warmup_path, &file_size));
  std::string file_content;
  file_content.resize(file_size);

  absl::string_view result;
  TF_RETURN_IF_ERROR(
      file->Read(0, result, absl::MakeSpan(&file_content[0], file_size)));
  if (result.size() < file_size) {
    return absl::DataLossError("Truncated read of warmup file");
  }

  google::protobuf::io::ArrayInputStream array_input(result.data(), result.size());

  neptune::RunOptions run_options;  // Default options, no deadline.
  bool clean_eof = false;
  neptune::serving::PredictionLog prediction_log;
  // TODO: Evaluate if parallelizing the warmup can speed up
  // server startup.
  while (google::protobuf::util::ParseDelimitedFromZeroCopyStream(
      &prediction_log, &array_input, &clean_eof)) {
    if (!prediction_log.has_predict_log()) {
      LOG(WARNING) << "PredictionLog does not have predict_log, skipping.";
      continue;
    }
    const neptune::serving::PredictRequest& request =
        prediction_log.predict_log().request();
    absl::string_view pipeline_name =
        request.model_spec().signature_name().empty()
            ? absl::string_view(kDefaultServingSignatureDefKey)
            : absl::string_view(request.model_spec().signature_name());
    absl::StatusOr<absl::flat_hash_map<std::string, neptune::Value>> warmup_status =
        RunPredictInternal(request, pipeline_name, run_options);
    if (!warmup_status.ok()) {
      LOG(ERROR) << "Warmup failed for pipeline '" << pipeline_name
                 << "': " << warmup_status.status();
      return warmup_status.status();
    }
    prediction_log.Clear();
  }

  if (!clean_eof) {
    return absl::DataLossError(
        "Failed to parse warmup file (truncated or corrupt)");
  }

  LOG(INFO) << "Warmup completed successfully.";
  return absl::OkStatus();
}

grpc::ServerUnaryReactor* OrbaxCallbackGrpcServer::GetModelMetadata(
    grpc::CallbackServerContext* context,
    const neptune::serving::GetModelMetadataRequest* request,
    neptune::serving::GetModelMetadataResponse* response) {
  VLOG(4) << "Received GetModelMetadataRequest: " << request->DebugString();
  grpc::ServerUnaryReactor* reactor = context->DefaultReactor();

  if (interpreter_ == nullptr) {
    reactor->Finish(grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                                 "Interpreter not initialized."));
    return reactor;
  }

  if (request->model_spec().name() != interpreter_->GetModelName()) {
    reactor->Finish(grpc::Status(
        grpc::StatusCode::FAILED_PRECONDITION,
        absl::StrCat("Model name mismatch: got [", request->model_spec().name(),
                     "] but expected [", interpreter_->GetModelName(), "]")));
    return reactor;
  }

  absl::string_view signature_name =
      request->model_spec().signature_name().empty()
          ? absl::string_view(kDefaultServingSignatureDefKey)
          : absl::string_view(request->model_spec().signature_name());

  auto pipeline_or = interpreter_->GetPipelineProto(signature_name);
  if (!pipeline_or.ok()) {
    reactor->Finish(ToGrpcStatus(pipeline_or.status()));
    return reactor;
  }
  const auto& pipeline = pipeline_or.value();

  response->mutable_model_spec()->CopyFrom(request->model_spec());

  for (const auto& field : request->metadata_field()) {
    if (field == "signature_def") {
      neptune::serving::SignatureDefMap signature_def_map;
      auto& sig_def =
          (*signature_def_map
                .mutable_signature_def())[std::string(signature_name)];
      *sig_def.mutable_signature() = pipeline.signature();
      if (pipeline.has_named_signature()) {
        *sig_def.mutable_named_signature() = pipeline.named_signature();
      }

      google::protobuf::Any any;
      any.PackFrom(signature_def_map);
      (*response->mutable_metadata())["signature_def"] = any;
    }
  }

  reactor->Finish(grpc::Status::OK);
  return reactor;
}

}  // namespace serving
}  // namespace neptune
