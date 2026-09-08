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

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/flags/flag.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "include/nanobind/nanobind.h"  // from @nanobind_bazel
#include "include/nanobind/nb_defs.h"  // from @nanobind_bazel
#include "include/nanobind/stl/shared_ptr.h"  // from @nanobind_bazel  // IWYU pragma: keep
#include "include/nanobind/stl/string.h"  // from @nanobind_bazel  // IWYU pragma: keep
#include "include/nanobind/stl/string_view.h"  // from @nanobind_bazel  // IWYU pragma: keep
#include "include/nanobind/stl/unordered_map.h"  // from @nanobind_bazel  // IWYU pragma: keep
#include "include/nanobind/stl/vector.h"  // from @nanobind_bazel  // IWYU pragma: keep
#include "runtime/core/device_buffer.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune
#include "runtime/core/value.h"  // from @neptune
#include "runtime/interpreter/interpreter.h"  // from @neptune
#include "runtime/interpreter/run_options.h"  // from @neptune
#include "runtime/kernel/kernel_io_data_type.h"  // from @neptune
#include "runtime/python/ifrt_client_factory.h"  // from @neptune
#include "runtime/python/numpy_utils.h"  // from @neptune
#include "runtime/tensorflow/tf_compatible_interpreter_create_options.h"  // from @neptune
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"  // from @xla
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/nb_numpy.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_client.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "xla/tsl/python/lib/core/numpy.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla
#include "tensorflow/core/framework/tensor.h"  // from @org_tensorflow
#include "tensorflow/core/framework/tensor.pb.h"  // from @org_tensorflow
#include "tensorflow/core/framework/tensor_shape.h"  // from @org_tensorflow
#include "tensorflow/core/framework/types.pb.h"  // from @org_tensorflow

namespace nb = ::nanobind;

namespace {

// Throws a `nb::value_error` with the given message.
void ThrowIfError(const absl::Status& status, absl::string_view msg) {
  if (status.ok()) return;
  throw nb::value_error(
      absl::StrCat(msg, ". Error: ", status.message()).c_str());
}

// Converts a `xla::nb_numpy_ndarray` to a `neptune::Value` for input.
neptune::Value GetValueFromNbArray(const xla::nb_numpy_ndarray& array,
                               neptune::KernelIODataType data_type) {
  switch (data_type) {
    case neptune::KernelIODataType::kHostBuffer:
      return neptune::Value(neptune::ToHostBuffer(array));
    case neptune::KernelIODataType::kTensor:
      return neptune::Value(neptune::ToTFTensor(array));
    default:
      throw nb::value_error(
          absl::StrCat("Unsupported data type: ", static_cast<int>(data_type))
              .c_str());
  }
}

// Converts a `neptune::Value` to a `xla::nb_numpy_ndarray` for output.
xla::nb_numpy_ndarray GetNbArrayFromValue(const neptune::Value& value,
                                          neptune::KernelIODataType data_type) {
  if (value.IsType<neptune::DeviceBuffer>()) {
    throw nb::value_error(
        "Received on-device buffer (neptune::DeviceBuffer) in Python layer. "
        "Host transfer (D2H) must be requested via Placement::kHost.");
  }
  switch (data_type) {
    case neptune::KernelIODataType::kHostBuffer:
      if (!value.IsType<neptune::HostBuffer>()) {
        throw nb::value_error("Value type mismatch: expected neptune::HostBuffer.");
      }
      return neptune::ToNbArray(value.Get<neptune::HostBuffer>());
    case neptune::KernelIODataType::kTensor:
      if (!value.IsType<tensorflow::Tensor>()) {
        throw nb::value_error(
            "Value type mismatch: expected tensorflow::Tensor.");
      }
      return neptune::ToNbArray(value.Get<tensorflow::Tensor>());
    default:
      throw nb::value_error(
          absl::StrCat("Unsupported data type: ", static_cast<int>(data_type))
              .c_str());
  }
}

// Converts args-kwargs to a list of positional inputs and a map of keyword
// inputs.
std::pair<std::vector<neptune::Value>, absl::flat_hash_map<std::string, neptune::Value>>
ConvertArgsKwargsToInputs(nb::args args, nb::kwargs kwargs,
                          neptune::KernelIODataType data_type) {
  std::vector<neptune::Value> positional_inputs;
  for (const auto& arg : args) {
    auto array = xla::nb_numpy_ndarray::ensure(arg);
    positional_inputs.push_back(GetValueFromNbArray(array, data_type));
  }

  absl::flat_hash_map<std::string, neptune::Value> keyword_inputs;
  for (const auto& [key, arg] : kwargs) {
    std::string key_str = nb::cast<std::string>(key, true);
    if (key_str == "run_options") {
      continue;
    }
    auto array = xla::nb_numpy_ndarray::ensure(arg);
    keyword_inputs[key_str] = GetValueFromNbArray(array, data_type);
  }
  return {std::move(positional_inputs), std::move(keyword_inputs)};
}

// ModelRunner is a wrapper of `neptune::Interpreter` exposed to Python for
// model loading and inference on the python side.
// Use `std::enable_shared_from_this` to keep the ModelRunner alive when
// `Signature` is beingused.
class ModelRunner : public std::enable_shared_from_this<ModelRunner> {
 public:
  static std::shared_ptr<ModelRunner> Create(
      absl::string_view model_name, absl::string_view model_path,
      absl::string_view platform = "",
      absl::string_view target_topology_label = "",
      bool allow_jit_fallback_for_aot = false) {
    return std::shared_ptr<ModelRunner>(
        new ModelRunner(model_name, model_path, platform, target_topology_label,
                        allow_jit_fallback_for_aot));
  }

  // Signature is a wrapper of `neptune::Interpreter` exposed to Python for
  // running a single pipeline in the model. Each signature is associated with
  // a pipeline in the model.
  class Signature {
   public:
    // Runs the model with given inputs. Each arg or kwarg must be convertible
    // to a `xla::nb_numpy_ndarray`.
    std::vector<xla::nb_numpy_ndarray> Run(nb::args args, nb::kwargs kwargs);

    // Same as `Run`, but returns a map of output values. Unlike `Run`, this
    // version can only be used when the model's output names are available.
    std::unordered_map<std::string, xla::nb_numpy_ndarray> RunWithMapReturn(
        nb::args args, nb::kwargs kwargs);

   private:
    // Use `ModelRunner` as a friend class to access the private constructor.
    // This makes the `Signature` usable only by `ModelRunner`.
    friend class ModelRunner;
    Signature(std::string pipeline_name, neptune::Interpreter* interpreter,
              std::shared_ptr<ModelRunner> model_runner)
        : pipeline_name_(std::move(pipeline_name)),
          interpreter_(interpreter),
          model_runner_(std::move(model_runner)) {}
    std::string pipeline_name_;
    neptune::Interpreter* absl_nonnull interpreter_;  // Not owned.
    std::shared_ptr<ModelRunner> model_runner_;  // Keep the model runner alive.
  };

  // Returns the platform name of the IFRT client.
  absl::string_view IfrtPlatformName() const {
    return client_->platform_name();
  }

  // Returns the device count of the IFRT client.
  int IfrtDeviceCount() const { return client_->device_count(); }

  // Returns the names of the signatures(pipelines) in the model.
  std::vector<std::string> SignatureNames() const {
    return interpreter_->GetPipelineNames();
  }

  // Returns a `Signature` for the given name.
  Signature GetSignature(absl::string_view signature_name);

 private:
  ModelRunner(absl::string_view model_name, absl::string_view model_path,
              absl::string_view platform,
              absl::string_view target_topology_label,
              bool allow_jit_fallback_for_aot) {
    LOG(INFO) << "Using ModelRunner to load model '" << model_name
              << "' from: " << model_path;
    ThrowIfError(Initialize(model_name, model_path, platform,
                            target_topology_label, allow_jit_fallback_for_aot),
                 "Failed to initialize ModelRunner");
  }

  absl::Status Initialize(absl::string_view model_name,
                          absl::string_view model_path,
                          absl::string_view platform,
                          absl::string_view target_topology_label,
                          bool allow_jit_fallback_for_aot);
  static constexpr int kThreadPoolNumThreads = 16;
  std::unique_ptr<tsl::thread::ThreadPool> execute_worker_thread_pool_;
  std::shared_ptr<xla::ifrt::Client> client_;
  std::shared_ptr<xla::ifrt::Client> host_ifrt_client_;
  std::unique_ptr<tsl::thread::ThreadPool> computation_thread_pool_;
  std::unique_ptr<tsl::thread::ThreadPool> checkpoint_loader_thread_pool_;

  std::unique_ptr<neptune::Interpreter> interpreter_;
};

absl::Status ModelRunner::Initialize(absl::string_view model_name,
                                     absl::string_view model_path,
                                     absl::string_view platform,
                                     absl::string_view target_topology_label,
                                     bool allow_jit_fallback_for_aot) {
  std::string platform_str(platform);
  if (platform_str.empty()) {
    platform_str = "cpu";
  }
  TF_ASSIGN_OR_RETURN(client_, neptune::CreateIfrtClient(platform_str));
  if (platform_str == "cpu") {
    host_ifrt_client_ = client_;
  } else {
    TF_ASSIGN_OR_RETURN(auto pjrt_client,
                        xla::GetXlaPjrtCpuClient(xla::CpuClientOptions()));
    host_ifrt_client_ = xla::ifrt::PjRtClient::Create(std::move(pjrt_client));
  }
  LOG(INFO) << "Platform: " << platform_str
            << ". Device count: " << client_->device_count();

  execute_worker_thread_pool_ = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(),
      "ModelRunnerExecutionThreadPool", kThreadPoolNumThreads);
  computation_thread_pool_ = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(),
      "ModelRunnerComputationThreadPool", kThreadPoolNumThreads);
  checkpoint_loader_thread_pool_ = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(),
      "ModelRunnerCheckpointLoaderThreadPool", kThreadPoolNumThreads);
  // Support model version in ModelRunner if we see use cases.
  auto create_options =
      std::make_unique<neptune::TfCompatibleInterpreterCreateOptions>();
  create_options->checkpoint_loader_thread_pool =
      checkpoint_loader_thread_pool_.get();
  create_options->model_name = model_name;
  create_options->model_version = 0;
  create_options->model_path = model_path;
  create_options->execute_worker_thread_pool =
      execute_worker_thread_pool_.get();
  create_options->ifrt_client = client_;
  create_options->host_ifrt_client = host_ifrt_client_;
  create_options->platform_name = platform_str;
  create_options->computation_thread_pool = computation_thread_pool_.get();
  create_options->target_topology_label = std::string(target_topology_label);
  create_options->allow_jit_fallback_for_aot = allow_jit_fallback_for_aot;
  TF_ASSIGN_OR_RETURN(interpreter_,
                      neptune::Interpreter::Create(std::move(create_options)));
  return absl::OkStatus();
}

ModelRunner::Signature ModelRunner::GetSignature(
    absl::string_view signature_name) {
  std::vector<std::string> signature_names = SignatureNames();
  if (std::find(signature_names.begin(), signature_names.end(),
                signature_name) == signature_names.end()) {
    throw nb::value_error(absl::StrCat("No signature named \"", signature_name,
                                       "\" found in the model. Available "
                                       "signatures: ",
                                       absl::StrJoin(signature_names, ", "))
                              .c_str());
  }
  return Signature(std::string(signature_name), interpreter_.get(),
                   shared_from_this());
}

std::vector<xla::nb_numpy_ndarray> ModelRunner::Signature::Run(
    nb::args args, nb::kwargs kwargs) {
  if (interpreter_ == nullptr) {
    throw nb::value_error(
        "The interpreter is not initialized. Check if the model is correctly "
        "loaded.");
  }
  neptune::RunOptions run_options;
  if (kwargs.contains("run_options") && !kwargs["run_options"].is_none()) {
    run_options = nb::cast<neptune::RunOptions>(kwargs["run_options"]);
  }
  auto input_type = interpreter_->GetModelInputDataType(pipeline_name_);
  ThrowIfError(input_type.status(), "Failed to get model input data type");
  auto [positional_inputs, keyword_inputs] =
      ConvertArgsKwargsToInputs(args, kwargs, *input_type);

  absl::StatusOr<std::vector<neptune::Value>> outputs = interpreter_->Run(
      pipeline_name_, positional_inputs, keyword_inputs, run_options);
  ThrowIfError(outputs.status(), "Failed to run the model");
  std::vector<neptune::Value> output_values = *std::move(outputs);
  auto output_type = interpreter_->GetModelOutputDataType(pipeline_name_);
  ThrowIfError(output_type.status(), "Failed to get model output data type");
  std::vector<xla::nb_numpy_ndarray> output_arrays;
  output_arrays.reserve(output_values.size());
  for (const auto& output_value : output_values) {
    output_arrays.push_back(GetNbArrayFromValue(output_value, *output_type));
  }

  return output_arrays;
}

std::unordered_map<std::string, xla::nb_numpy_ndarray>
ModelRunner::Signature::RunWithMapReturn(nb::args args, nb::kwargs kwargs) {
  if (interpreter_ == nullptr) {
    throw nb::value_error(
        "The interpreter is not initialized. This should never happen.");
  }
  neptune::RunOptions run_options;
  if (kwargs.contains("run_options") && !kwargs["run_options"].is_none()) {
    run_options = nb::cast<neptune::RunOptions>(kwargs["run_options"]);
  }
  auto input_type = interpreter_->GetModelInputDataType(pipeline_name_);
  ThrowIfError(input_type.status(), "Failed to get model input data type");
  auto [positional_inputs, keyword_inputs] =
      ConvertArgsKwargsToInputs(args, kwargs, *input_type);
  absl::StatusOr<absl::flat_hash_map<std::string, neptune::Value>> outputs =
      interpreter_->RunWithMapReturn(pipeline_name_, positional_inputs,
                                     keyword_inputs, run_options);
  ThrowIfError(outputs.status(), "Failed to run the model");
  absl::flat_hash_map<std::string, neptune::Value> output_values =
      *std::move(outputs);
  auto output_type = interpreter_->GetModelOutputDataType(pipeline_name_);
  ThrowIfError(output_type.status(), "Failed to get model output data type");
  std::unordered_map<std::string, xla::nb_numpy_ndarray> output_arrays;
  for (const auto& [name, value] : output_values) {
    output_arrays[name] = GetNbArrayFromValue(value, *output_type);
  }

  return output_arrays;
}

}  // namespace

NB_MODULE(model_runner, m) {
  tsl::ImportNumpy();

  nb::enum_<neptune::Placement>(m, "Placement")
      .value("HOST", neptune::Placement::kHost)
      .value("DEVICE", neptune::Placement::kDevice)
      .export_values();

  nb::class_<neptune::RunOptions>(m, "RunOptions")
      .def(nb::init<>())
      .def_prop_rw(
          "output_placement_map",
          [](const neptune::RunOptions& self) {
            return std::unordered_map<std::string, neptune::Placement>(
                self.output_placement_map.begin(),
                self.output_placement_map.end());
          },
          [](neptune::RunOptions& self,
             const std::unordered_map<std::string, neptune::Placement>& map) {
            self.output_placement_map = {map.begin(), map.end()};
          });

  nb::class_<ModelRunner>(m, "ModelRunner")
      .def(nb::new_(&ModelRunner::Create),
           nb::arg("model_name") = "called_from_python", nb::arg("model_path"),
           nb::arg("platform") = "", nb::arg("target_topology_label") = "",
           nb::arg("allow_jit_fallback_for_aot") = false)
      .def("ifrt_platform_name", &ModelRunner::IfrtPlatformName,
           "Gets the platform name (e.g. 'cpu', 'tpu') of the IFRT client.")
      .def("ifrt_device_count", &ModelRunner::IfrtDeviceCount,
           "Gets the device count of the IFRT client.")
      .def("signature_names", &ModelRunner::SignatureNames,
           "Gets the names of the available signatures in the model.")
      .def("signature", &ModelRunner::GetSignature,
           "Gets the signature with the given name.");

  nb::class_<ModelRunner::Signature>(m, "Signature")
      .def("run", &ModelRunner::Signature::Run,
           "Runs the model with given args and kwargs. Each arg and kwarg must "
           "represent a numpy array-like object (e.g., `numpy.ndarray`, "
           "`tf.Tensor`, `jax.Array`). Returns a list of numpy arrays as "
           "outputs. Example usage:"
           "```python"
           "signature = model_runner.Signature('name')"
           "signature.run(np.array([1, 2]), np.array([3, 4]))"
           "signature.run(a=np.array([1, 2]), b=np.array([3, 4]))"
           "```")
      .def("run_with_dict_return", &ModelRunner::Signature::RunWithMapReturn,
           "Runs the model with given args and kwargs. Same as `run`, but "
           "returns a dict of string to numpy arrays as outputs.");
}
