#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "learning/infra/mira/experimental/orbax_model/interpreter/interpreter.h"
#include "learning/infra/mira/experimental/orbax_model/tensorflow/tf_compatible_interpreter_create_options.h"
#include "third_party/absl/base/nullability.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/flags/flag.h"
#include "third_party/absl/log/log.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/str_join.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "third_party/nanobind/include/nanobind/nb_defs.h"
#include "third_party/nanobind/include/nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/string.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/unordered_map.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/vector.h"  // IWYU pragma: keep
#include "third_party/neptune/runtime/core/host_buffer.h"
#include "third_party/neptune/runtime/core/value.h"
#include "third_party/neptune/runtime/kernel/kernel_io_data_type.h"
#include "third_party/neptune/runtime/python/ifrt_client_factory.h"
#include "third_party/neptune/runtime/python/numpy_utils.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/client.h"
#include "third_party/tensorflow/compiler/xla/python/nb_numpy.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/env.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/threadpool.h"
#include "third_party/tensorflow/compiler/xla/tsl/python/lib/core/numpy.h"
#include "third_party/tensorflow/compiler/xla/xla_data.proto.h"
#include "third_party/tensorflow/core/framework/tensor.h"
#include "third_party/tensorflow/core/framework/tensor.proto.h"
#include "third_party/tensorflow/core/framework/tensor_shape.h"
#include "third_party/tensorflow/core/framework/types.proto.h"

ABSL_FLAG(std::string, orbax_model_runner_platform, "cpu",
          "The platform to use. It can be cpu, tpu, or cuda");

namespace nb = ::nanobind;

namespace {

// Throws a `nb::value_error` with the given message.
void ThrowIfError(const absl::Status& status, absl::string_view msg) {
  if (status.ok()) return;
  throw nb::value_error(
      absl::StrCat(msg, ". Error: ", status.message()).c_str());
}

// Converts a `xla::nb_numpy_ndarray` to a `jsv::Value` for input.
jsv::Value GetValueFromNbArray(const xla::nb_numpy_ndarray& array,
                               jsv::KernelIODataType data_type) {
  switch (data_type) {
    case jsv::KernelIODataType::kHostBuffer:
      return jsv::Value(neptune::ToHostBuffer(array));
    case jsv::KernelIODataType::kTensor:
      return jsv::Value(neptune::ToTFTensor(array));
    default:
      throw nb::value_error(
          absl::StrCat("Unsupported data type: ", static_cast<int>(data_type))
              .c_str());
  }
}

// Converts a `jsv::Value` to a `xla::nb_numpy_ndarray` for output.
xla::nb_numpy_ndarray GetNbArrayFromValue(const jsv::Value& value,
                                          jsv::KernelIODataType data_type) {
  switch (data_type) {
    case jsv::KernelIODataType::kHostBuffer:
      return neptune::ToNbArray(value.Get<jsv::HostBuffer>());
    case jsv::KernelIODataType::kTensor:
      return neptune::ToNbArray(value.Get<tensorflow::Tensor>());
    default:
      throw nb::value_error(
          absl::StrCat("Unsupported data type: ", static_cast<int>(data_type))
              .c_str());
  }
}

// Converts args-kwargs to a list of positional inputs and a map of keyword
// inputs.
std::pair<std::vector<jsv::Value>, absl::flat_hash_map<std::string, jsv::Value>>
ConvertArgsKwargsToInputs(nb::args args, nb::kwargs kwargs,
                          jsv::KernelIODataType data_type) {
  std::vector<jsv::Value> positional_inputs;
  for (const auto& arg : args) {
    auto array = xla::nb_numpy_ndarray::ensure(arg);
    positional_inputs.push_back(GetValueFromNbArray(array, data_type));
  }

  absl::flat_hash_map<std::string, jsv::Value> keyword_inputs;
  for (const auto& [key, arg] : kwargs) {
    auto array = xla::nb_numpy_ndarray::ensure(arg);
    keyword_inputs[nb::cast<std::string>(key, true)] =
        GetValueFromNbArray(array, data_type);
  }
  return {std::move(positional_inputs), std::move(keyword_inputs)};
}

// ModelRunner is a wrapper of `jsv::Interpreter` exposed to Python for
// model loading and inference on the python side.
// Use `std::enable_shared_from_this` to keep the ModelRunner alive when
// `Signature` is beingused.
class ModelRunner : public std::enable_shared_from_this<ModelRunner> {
 public:
  static std::shared_ptr<ModelRunner> Create(absl::string_view model_name,
                                             absl::string_view model_path) {
    return std::shared_ptr<ModelRunner>(
        new ModelRunner(model_name, model_path));
  }

  // Signature is a wrapper of `jsv::Interpreter` exposed to Python for
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
    Signature(std::string pipeline_name, jsv::Interpreter* interpreter,
              std::shared_ptr<ModelRunner> model_runner)
        : pipeline_name_(std::move(pipeline_name)),
          interpreter_(interpreter),
          model_runner_(std::move(model_runner)) {}
    std::string pipeline_name_;
    jsv::Interpreter* absl_nonnull interpreter_;  // Not owned.
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
  ModelRunner(absl::string_view model_name, absl::string_view model_path) {
    LOG(INFO) << "Using ModelRunner to load Orbax model '" << model_name
              << "' from: " << model_path;
    ThrowIfError(Initialize(model_name, model_path),
                 "Failed to initialize ModelRunner");
  }

  absl::Status Initialize(absl::string_view model_name,
                          absl::string_view model_path);
  static constexpr int kThreadPoolNumThreads = 16;
  std::unique_ptr<tsl::thread::ThreadPool> execute_worker_thread_pool_;
  std::shared_ptr<xla::ifrt::Client> client_;
  std::unique_ptr<tsl::thread::ThreadPool> computation_thread_pool_;
  std::unique_ptr<tsl::thread::ThreadPool> checkpoint_loader_thread_pool_;

  std::unique_ptr<jsv::Interpreter> interpreter_;
};

absl::Status ModelRunner::Initialize(absl::string_view model_name,
                                     absl::string_view model_path) {
  std::string platform = absl::GetFlag(FLAGS_orbax_model_runner_platform);
  TF_ASSIGN_OR_RETURN(client_, neptune::CreateIfrtClient(platform));
  LOG(INFO) << "Platform: " << platform
            << ". Device count: " << client_->device_count();

  execute_worker_thread_pool_ = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "OrbaxExecutionThreadPool",
      kThreadPoolNumThreads);
  computation_thread_pool_ = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "OrbaxComputationThreadPool",
      kThreadPoolNumThreads);
  checkpoint_loader_thread_pool_ = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(),
      "OrbaxCheckpointLoaderThreadPool", kThreadPoolNumThreads);
  // Support model version in ModelRunner if we see use cases.
  auto create_options =
      std::make_unique<jsv::TfCompatibleInterpreterCreateOptions>();
  create_options->checkpoint_loader_thread_pool =
      checkpoint_loader_thread_pool_.get();
  create_options->model_name = model_name;
  create_options->model_version = 0;
  create_options->model_path = model_path;
  create_options->execute_worker_thread_pool =
      execute_worker_thread_pool_.get();
  create_options->ifrt_client = client_;
  create_options->platform_name = platform;
  create_options->computation_thread_pool = computation_thread_pool_.get();
  TF_ASSIGN_OR_RETURN(interpreter_,
                      jsv::Interpreter::Create(std::move(create_options)));
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
  auto [positional_inputs, keyword_inputs] = ConvertArgsKwargsToInputs(
      args, kwargs, *interpreter_->GetModelInputDataType(pipeline_name_));

  absl::StatusOr<std::vector<jsv::Value>> outputs =
      interpreter_->Run(pipeline_name_, positional_inputs, keyword_inputs);
  ThrowIfError(outputs.status(), "Failed to run the model");
  std::vector<jsv::Value> output_values = *std::move(outputs);
  std::vector<xla::nb_numpy_ndarray> output_arrays;
  output_arrays.reserve(output_values.size());
  for (const auto& output_value : output_values) {
    output_arrays.push_back(GetNbArrayFromValue(
        output_value, *interpreter_->GetModelOutputDataType(pipeline_name_)));
  }

  return output_arrays;
}

std::unordered_map<std::string, xla::nb_numpy_ndarray>
ModelRunner::Signature::RunWithMapReturn(nb::args args, nb::kwargs kwargs) {
  if (interpreter_ == nullptr) {
    throw nb::value_error(
        "The interpreter is not initialized. This should never happen.");
  }
  auto [positional_inputs, keyword_inputs] = ConvertArgsKwargsToInputs(
      args, kwargs, *interpreter_->GetModelInputDataType(pipeline_name_));
  absl::StatusOr<absl::flat_hash_map<std::string, jsv::Value>> outputs =
      interpreter_->RunWithMapReturn(pipeline_name_, positional_inputs,
                                     keyword_inputs);
  ThrowIfError(outputs.status(), "Failed to run the model");
  absl::flat_hash_map<std::string, jsv::Value> output_values =
      *std::move(outputs);
  std::unordered_map<std::string, xla::nb_numpy_ndarray> output_arrays;
  for (const auto& [name, value] : output_values) {
    output_arrays[name] = GetNbArrayFromValue(
        value,
        *interpreter_->GetModelOutputDataType(pipeline_name_));
  }

  return output_arrays;
}

}  // namespace

NB_MODULE(orbax_model_runner, m) {
  nb::class_<ModelRunner>(m, "ModelRunner")
      .def(nb::new_(&ModelRunner::Create),
           nb::arg("model_name") = "called_from_python", nb::arg("model_path"))
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
