#include "third_party/neptune/runtime/serving/gcp/server/orbax_grpc_server.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "learning/infra/mira/experimental/orbax_model/interpreter/interpreter.h"
#include "learning/infra/mira/experimental/orbax_model/interpreter/run_options.h"
#include "learning/infra/mira/experimental/orbax_model/tensorflow/tf_compatible_interpreter_create_options.h"
#include "learning/infra/mira/experimental/orbax_model/tensorflow/utils/tf_to_host_buffer.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/flags/flag.h"
#include "third_party/absl/log/log.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/str_join.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/time/time.h"
#include "third_party/grpc/include/grpcpp/server_context.h"
#include "third_party/grpc/include/grpcpp/support/status.h"
#include "third_party/neptune/runtime/core/host_buffer.h"
#include "third_party/neptune/runtime/core/proto/host_buffer.proto.h"
#include "third_party/neptune/runtime/core/value.h"
#include "third_party/neptune/runtime/kernel/kernel_io_data_type.h"
#include "third_party/neptune/runtime/serving/gcp/proto/model.proto.h"
#include "third_party/neptune/runtime/serving/gcp/proto/prediction_service.proto.h"
#include "third_party/tensorflow/compiler/xla/pjrt/pjrt_api.h"
#include "third_party/tensorflow/compiler/xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "third_party/tensorflow/compiler/xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "third_party/tensorflow/compiler/xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "third_party/tensorflow/compiler/xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.h"
#include "third_party/tensorflow/compiler/xla/pjrt/plugin/xla_tpu/xla_tpu_pjrt_client.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/client.h"
#include "third_party/tensorflow/compiler/xla/python/pjrt_ifrt/pjrt_client.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/env.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/errors.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/threadpool.h"
#include "third_party/tensorflow/core/framework/tensor.h"

ABSL_FLAG(std::string, orbax_grpc_server_platform, "cpu",
          "The platform to use. It can be cpu, tpu, or cuda.");

namespace jsv {
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
  auto create_options =
      std::make_unique<jsv::TfCompatibleInterpreterCreateOptions>();
  create_options->checkpoint_loader_thread_pool =
      checkpoint_loader_thread_pool.get();
  create_options->model_name = model_name;
  create_options->model_version = 0;
  create_options->model_path = model_path;
  create_options->execute_worker_thread_pool = execute_worker_thread_pool.get();
  create_options->ifrt_client = client;
  create_options->computation_thread_pool = computation_thread_pool.get();
  create_options->platform_name = orbax_platform;
  std::unique_ptr<jsv::Interpreter> interpreter;
  TF_ASSIGN_OR_RETURN(interpreter,
                      jsv::Interpreter::Create(std::move(create_options)));
  return std::make_unique<OrbaxGrpcServer>(
      model_name, std::move(interpreter), client,
      std::move(execute_worker_thread_pool), std::move(computation_thread_pool),
      std::move(checkpoint_loader_thread_pool));
}

grpc::Status OrbaxGrpcServer::Predict(
    grpc::ServerContext* context, const jsv::serving::PredictRequest* request,
    jsv::serving::PredictResponse* response) {
  VLOG(4) << "Received PredictRequest: " << request->DebugString();
  if (interpreter_ == nullptr) {
    return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                        "Interpreter not initialized.");
  }

  std::string_view pipeline_name =
      request->model_spec().signature_name().empty()
          ? kDefaultServingSignatureDefKey
          : request->model_spec().signature_name();
  VLOG(1) << "Using pipeline name: '" << pipeline_name << "'";

  TF_ASSIGN_OR_RETURN(jsv::KernelIODataType input_data_type,
                      interpreter_->GetModelInputDataType(pipeline_name));
  bool input_expects_tensor =
      (input_data_type == jsv::KernelIODataType::kTensor);
  VLOG(3) << "Input expects tensor: " << input_expects_tensor;

  absl::flat_hash_map<std::string, jsv::Value> input_with_names;
  for (const auto& [name, host_buffer_proto] : request->inputs()) {
    TF_ASSIGN_OR_RETURN(jsv::HostBuffer host_buffer,
                        jsv::HostBuffer::FromProto(host_buffer_proto));
    if (input_expects_tensor) {
      TF_ASSIGN_OR_RETURN(tensorflow::Tensor tensor, ToTFTensor(host_buffer));
      input_with_names[name] = jsv::Value(std::move(tensor));
    } else {
      input_with_names[name] = jsv::Value(std::move(host_buffer));
    }
  }
  LogMapKeys(input_with_names, "Input with names", /*vlog_level=*/2);

  jsv::RunOptions run_options;
  run_options.deadline = absl::FromChrono(context->deadline());
  TF_ASSIGN_OR_RETURN(
      auto output_with_names,
      interpreter_->RunWithMapReturn(pipeline_name, /*positional_inputs=*/{},
                                     input_with_names, run_options));
  LogMapKeys(output_with_names, "Output with names", /*vlog_level=*/2);

  TF_ASSIGN_OR_RETURN(jsv::KernelIODataType output_data_type,
                      interpreter_->GetModelOutputDataType(pipeline_name));
  bool output_is_tensor = (output_data_type == jsv::KernelIODataType::kTensor);
  VLOG(3) << "Output is tensor: " << output_is_tensor;

  for (const auto& [name, value] : output_with_names) {
    jsv::HostBuffer host_buffer;
    if (output_is_tensor) {
      TF_ASSIGN_OR_RETURN(host_buffer,
                          ToHostBuffer(value.Get<tensorflow::Tensor>()));
    } else {
      host_buffer = value.Get<jsv::HostBuffer>();
    }
    HostBufferProto host_buffer_proto =
        HostBuffer::ToHostBufferProto(std::move(host_buffer));
    (*response->mutable_outputs())[name] = std::move(host_buffer_proto);
  }

  response->mutable_model_spec()->set_name(model_name_);
  response->mutable_model_spec()->set_signature_name(pipeline_name);

  return grpc::Status::OK;
}

grpc::Status OrbaxGrpcServer::GetModelMetadata(
    grpc::ServerContext* context,
    const jsv::serving::GetModelMetadataRequest* request,
    jsv::serving::GetModelMetadataResponse* response) {
  return grpc::Status(grpc::StatusCode::UNIMPLEMENTED, "Not implemented");
}

}  // namespace serving
}  // namespace jsv
