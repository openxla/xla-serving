#include "third_party/neptune/runtime/core/stablehlo_runner.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "learning/infra/mira/experimental/orbax_model/interpreter/manifest_utils.h"
#include "third_party/absl/base/call_once.h"
#include "third_party/absl/base/nullability.h"
#include "third_party/absl/base/optimization.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/container/flat_hash_set.h"
#include "third_party/absl/log/check.h"
#include "third_party/absl/log/log.h"
#include "third_party/absl/memory/memory.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/synchronization/mutex.h"
#include "third_party/absl/time/clock.h"
#include "third_party/absl/types/span.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/OwningOpRef.h"
#include "third_party/neptune/runtime/core/host_buffer.h"
#include "third_party/neptune/runtime/core/ifrt_device_utils.h"
#include "third_party/neptune/runtime/core/ifrt_loaded_variable_registry.h"
#include "third_party/neptune/runtime/core/loaded_stablehlo.h"
#include "third_party/neptune/runtime/core/sharding_utils.h"
#include "third_party/neptune/runtime/core/stablehlo.h"
#include "third_party/neptune/runtime/core/stablehlo_optimizer.h"
#include "third_party/neptune/runtime/core/stablehlo_utils.h"
#include "third_party/neptune/runtime/core/value.h"
#include "third_party/neptune/runtime/libs/ifrt/executable_build_utils.h"
#include "third_party/neptune/runtime/libs/ifrt/persistent_compilation_cache/ifrt_persistent_compilation_cache.h"
#include "third_party/neptune/runtime/libs/ifrt/serving_core_selector/serving_core_selector.h"
#include "third_party/neptune/runtime/libs/mlir/dump_mlir_util.h"
#include "third_party/protobuf/message.h"
#include "third_party/tensorflow/compiler/xla/hlo/ir/hlo_sharding.h"
#include "third_party/tensorflow/compiler/xla/layout.h"
#include "third_party/tensorflow/compiler/xla/pjrt/proto/compile_options.proto.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/array.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/client.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/device.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/device_list.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/dtype.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/executable.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/host_callback.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/shape.h"
#include "third_party/tensorflow/compiler/xla/python/refine_polymorphic_shapes.h"
#include "third_party/tensorflow/compiler/xla/service/computation_placer.h"
#include "third_party/tensorflow/compiler/xla/tsl/concurrency/future.h"
#include "third_party/tensorflow/compiler/xla/tsl/concurrency/ref_count.h"
#include "third_party/tensorflow/compiler/xla/tsl/framework/serving_device_selector.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/errors.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/threadpool.h"
#include "third_party/tensorflow/compiler/xla/xla_data.proto.h"
#include "third_party/tensorflow/tsl/profiler/lib/traceme.h"

namespace jsv {
namespace {

// Returns the device assignment from the given IFRT devices list.
absl::StatusOr<xla::DeviceAssignment> GetRuntimeXlaDeviceAssignment(
    const xla::ifrt::DeviceListRef& devices, int num_replicas,
    int num_cores_per_replica) {
  const int num_devices = num_replicas * num_cores_per_replica;
  if (devices->size() != num_devices) {
    return absl::FailedPreconditionError(
        absl::StrCat("Device assignment has ", devices->size(),
                     " devices, but expected ", num_devices));
  }
  xla::DeviceAssignment da(num_replicas, num_cores_per_replica);
  int device_index = 0;
  for (int replica_idx = 0; replica_idx < num_replicas; replica_idx++) {
    for (int core_idx = 0; core_idx < num_cores_per_replica;
         core_idx++, device_index++) {
      da(replica_idx, core_idx) =
          devices->devices()[device_index]->Id().value();
    }
  }
  return da;
}

}  // namespace

absl::StatusOr<std::vector<xla::ifrt::ArrayRef>>
StableHloRunner::DataConverter::ConvertInput(
    absl::Span<const HostBuffer> inputs, const StableHloIOSpec& io_spec,
    int input_offset) {
  tsl::profiler::TraceMe trace_me([&]() {
    return tsl::profiler::TraceMeEncode("TransferFromHostToDevice", {});
  });
  std::vector<xla::ifrt::ArrayRef> converted_inputs;
  converted_inputs.reserve(inputs.size());
  // Use a separate index because the io_spec contains both weights and
  // user inputs.
  int index = 0;
  for (int i = 0; i < inputs.size(); ++i) {
    const HostBuffer& input_data = inputs[i];
    const ArgSpec& input_spec = io_spec.inputs[index + input_offset];
    index++;
    if (input_data.dtype() != input_spec.dtype) {
      return absl::InvalidArgumentError(
          absl::StrCat("Input dtype mismatch. Expected: ", input_spec.dtype,
                       ", but got: ", input_data.dtype()));
    }
    const xla::ifrt::Shape& input_shape = input_data.shape();
    TF_RETURN_IF_ERROR(InputShapeEqual(input_spec.shape, input_shape));
    VLOG(2) << "Converting input of shape " << input_shape;
    TF_ASSIGN_OR_RETURN(xla::ifrt::ArrayRef array,
                        sharding_internal::MakeArrayFromHostBuffer(
                            *ifrt_client_, input_data, device_list_,
                            input_spec.sharding, *thread_pool_));

    converted_inputs.push_back(std::move(array));
  }
  return converted_inputs;
}

absl::StatusOr<std::vector<HostBuffer>>
StableHloRunner::DataConverter::ConvertOutput(
    absl::Span<const xla::ifrt::ArrayRef> stablehlo_outputs,
    const StableHloIOSpec& io_spec) {
  tsl::profiler::TraceMe trace_me([&]() {
    return tsl::profiler::TraceMeEncode("TransferFromDeviceToHost", {});
  });
  std::vector<tsl::Future<HostBuffer>> output_futures;
  output_futures.reserve(stablehlo_outputs.size());

  for (int i = 0; i < stablehlo_outputs.size(); ++i) {
    const xla::ifrt::ArrayRef& array_for_copy = stablehlo_outputs[i];
    // IFRT's return does not contain sufficient information; so we use
    // sharding spec from metadata.
    VLOG(2) << "Output sharding: " << array_for_copy->sharding();
    output_futures.push_back(sharding_internal::MakeHostBufferFromArray(
        *ifrt_client_, *array_for_copy, io_spec.outputs[i].sharding,
        device_list_, *thread_pool_));
  }

  std::vector<HostBuffer> converted_outputs;
  converted_outputs.reserve(stablehlo_outputs.size());
  for (tsl::Future<HostBuffer>& output_future : output_futures) {
    TF_ASSIGN_OR_RETURN(auto host_buffer, output_future.Await());
    converted_outputs.push_back(std::move(host_buffer));
  }
  return converted_outputs;
}

StableHloRunner::StableHloRunner(
    absl_nonnull std::unique_ptr<LoadedStableHlo> loaded_stablehlo,
    std::optional<xla::CompileOptionsProto> compile_options,
    absl_nonnull std::shared_ptr<xla::ifrt::Client> client,
    IfrtLoadedVariableRegistry* absl_nonnull ifrt_loaded_variable_registry,
    std::vector<std::string> weights,
    xla::ifrt::DeviceListRef assigned_device_list,
    tsl::protobuf::Message* compilation_environment_proto,
    jsv::IfrtPersistentCompilationCache* absl_nullable
        persistent_compilation_cache,
    StablehloOptimizer* absl_nullable optimizer,
    absl_nonnull std::shared_ptr<const AssignedDevices> assigned_devices,
    jsv::ServingCoreSelector* absl_nullable serving_core_selector)
    : loaded_stablehlo_(std::move(loaded_stablehlo)),
      compile_options_(std::move(compile_options)),
      assigned_device_list_(std::move(assigned_device_list)),
      ifrt_client_(std::move(client)),
      persistent_compilation_cache_(persistent_compilation_cache),
      ifrt_loaded_variable_registry_(*ifrt_loaded_variable_registry),
      weights_(std::move(weights)),
      compilation_environment_proto_(compilation_environment_proto),
      optimizer_(optimizer),
      assigned_devices_(std::move(assigned_devices)),
      serving_core_selector_(serving_core_selector) {}

absl::StatusOr<std::unique_ptr<StableHloRunner>> StableHloRunner::Create(
    absl_nonnull std::shared_ptr<StableHlo> stablehlo,
    std::optional<xla::CompileOptionsProto> compile_options,
    absl_nonnull std::shared_ptr<xla::ifrt::Client> client,
    jsv::IfrtPersistentCompilationCache* absl_nullable
        persistent_compilation_cache,
    IfrtLoadedVariableRegistry* absl_nonnull ifrt_loaded_variable_registry,
    std::vector<std::string> weights,
    tsl::protobuf::Message* absl_nullable compilation_environment_proto,
    tsl::thread::ThreadPool* absl_nonnull thread_pool,
    StablehloOptimizer* absl_nullable optimizer,
    absl_nonnull std::shared_ptr<const AssignedDevices> assigned_devices,
    jsv::ServingCoreSelector* absl_nullable serving_core_selector) {
  if (!stablehlo) {
    return absl::InvalidArgumentError(
        "`StableHlo` should be non-null when creating `StableHloRunner`");
  }
  if (client == nullptr) {
    return absl::InvalidArgumentError(
        "`ifrt::Client` client should be non-null when creating "
        "`StableHloRunner`");
  }
  if (ifrt_loaded_variable_registry == nullptr) {
    return absl::InvalidArgumentError(
        "`IfrtLoadedVariableRegistry` should be non-null when creating "
        "`StableHloRunner`");
  }
  if (thread_pool == nullptr) {
    return absl::InvalidArgumentError(
        "Thread pool is null when creating `StableHloRunner`");
  }
  if (assigned_devices == nullptr) {
    return absl::InvalidArgumentError(
        "Assigned devices is null when creating `StableHloRunner`");
  }

  // Load StableHlo.
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<LoadedStableHlo> loaded_stablehlo,
      LoadedStableHlo::Create(std::move(stablehlo), weights.size(),
                              client->platform_name(), compile_options));

  TF_ASSIGN_OR_RETURN(
      std::vector<xla::ifrt::Device*> assigned_ifrt_devices,
      GetAssignedIfrtDevices(*client, assigned_devices->num_replicas,
                             assigned_devices->num_cores_per_replica,
                             assigned_devices->device_assignment));
  TF_ASSIGN_OR_RETURN(xla::ifrt::DeviceListRef assigned_device_list,
                      client->MakeDeviceList(assigned_ifrt_devices));

  auto runner = absl::WrapUnique(new StableHloRunner(
      std::move(loaded_stablehlo), std::move(compile_options),
      std::move(client), ifrt_loaded_variable_registry, std::move(weights),
      assigned_device_list, compilation_environment_proto,
      persistent_compilation_cache, optimizer, assigned_devices,
      serving_core_selector));
  TF_RETURN_IF_ERROR(runner->Init(thread_pool));
  return runner;
}

absl::Status StableHloRunner::Init(
    tsl::thread::ThreadPool* absl_nonnull thread_pool) {
  if (IsPortableExecutable()) {
    // Create resources for all devices.
    for (size_t i = 0; i < ifrt_client_->addressable_device_count(); ++i) {
      TF_ASSIGN_OR_RETURN(xla::ifrt::Device * device,
                          ifrt_client_->LookupDevice(xla::ifrt::DeviceId(i)));
      TF_ASSIGN_OR_RETURN(xla::ifrt::DeviceListRef device_list,
                          ifrt_client_->MakeDeviceList({device}));
      device_list_to_resources_map_[device_list] =
          std::make_unique<ResourcesPerDeviceList>(ifrt_client_, device_list,
                                                   thread_pool);
    }
  } else {
    device_list_to_resources_map_[assigned_device_list_] =
        std::make_unique<ResourcesPerDeviceList>(
            ifrt_client_, assigned_device_list_, thread_pool);
  }
  return absl::OkStatus();
}

absl::StatusOr<StableHloRunner::SharedCachedExecutableBundle>
StableHloRunner::CreateExecutableSynchronously(
    absl::Span<const xla::ifrt::Shape> input_shapes,
    absl::Span<const xla::ifrt::DType> input_dtypes,
    std::unique_ptr<LoadedStableHlo::ModuleBundle> module_bundle,
    const StablehloOptimizer::Args& args) {
  // Refine the dynamic shapes in the module.
  // Use any device list's weight arrays, because the weight specs are the
  // same across all device lists.
  // TODO: Pass weight specs instead of weight arrays to
  // `RefineShapesIfNeeded()`.
  if (device_list_to_resources_map_.empty()) {
    return absl::FailedPreconditionError(
        "No device list resources initialized.");
  }
  TF_RETURN_IF_ERROR(
      device_list_to_resources_map_.begin()->second->weight_arrays.status());
  std::vector<std::optional<xla::ifrt::ArrayRef>>* weight_arrays =
      &device_list_to_resources_map_.begin()->second->weight_arrays.value();
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      RefineShapesIfNeeded(module_bundle->module.get(), *weight_arrays,
                           input_shapes, input_dtypes,
                           module_bundle->module_kept_var_idx),
      "Failed to refine shapes.");
  TF_RETURN_IF_ERROR(xla::ValidateStaticShapes(module_bundle->module.get()));

  TF_ASSIGN_OR_RETURN(
      StableHloIOSpec io_spec,
      GetStableHloIOSpec(*module_bundle->module, compile_options_.has_value()
                                                     ? &compile_options_.value()
                                                     : nullptr));

  if (VLOG_IS_ON(1)) {
    jsv::DumpMlirOpToFile("stablehlo_module_after_refinement",
                          module_bundle->module.get());
  }

  if (optimizer_ != nullptr) {
    TF_RETURN_IF_ERROR(optimizer_->Optimize(args, module_bundle->module.get()));
  }

  if (VLOG_IS_ON(1)) {
    jsv::DumpMlirOpToFile("stablehlo_module_after_optimization",
                          module_bundle->module.get());
  }

  TF_ASSIGN_OR_RETURN(
      io_spec,
      GetStableHloIOSpec(*module_bundle->module, compile_options_.has_value()
                                                     ? &compile_options_.value()
                                                     : nullptr));

  jsv::BuildIfrtExecutableOptions options;

  if (compile_options_.has_value()) {
    options.compile_options = &*compile_options_;
  }
  // TODO: Remove these overrides once the compile options are
  // fully supported.
  options.num_replicas = assigned_devices_->num_replicas;
  options.num_cores_per_replica = assigned_devices_->num_cores_per_replica;
  if (IsPortableExecutable()) {
    options.compile_portable_executable = true;
  } else {
    TF_ASSIGN_OR_RETURN(
        xla::DeviceAssignment da,
        GetRuntimeXlaDeviceAssignment(
            assigned_device_list_, assigned_devices_->num_replicas,
            assigned_devices_->num_cores_per_replica));
    VLOG(2) << "Device assignment :" << da.ToString();
    options.device_assignment = std::move(da);
  }

  TF_ASSIGN_OR_RETURN(
      xla::ifrt::LoadedExecutableRef ifrt_executable,
      BuildIfrtExecutable(module_bundle->module.get(), options, ifrt_client_,
                          persistent_compilation_cache_));

  SharedCachedExecutableBundle executable_bundle =
      std::make_shared<CachedExecutableBundle>();
  executable_bundle->module_kept_var_idx =
      std::vector<int>(module_bundle->module_kept_var_idx.begin(),
                       module_bundle->module_kept_var_idx.end());
  executable_bundle->input_offset = module_bundle->input_offset;
  executable_bundle->io_spec = io_spec;
  executable_bundle->ifrt_executable = std::move(ifrt_executable);
  executable_bundle->program_fingerprint_with_timestamp =
      jsv::GetProgramFingerprintWithTimestamp(
          executable_bundle->ifrt_executable, absl::Now());

  return executable_bundle;
}

tsl::Future<StableHloRunner::SharedCachedExecutableBundle>
StableHloRunner::LookUpOrCreateExecutable(
    absl::Span<const xla::ifrt::Shape> input_shapes,
    absl::Span<const xla::ifrt::DType> input_dtypes) {
  tsl::Promise<SharedCachedExecutableBundle> promise;
  tsl::Future<SharedCachedExecutableBundle> future;
  std::unique_ptr<LoadedStableHlo::ModuleBundle> module_bundle;

  {
    absl::MutexLock lock(mutex_);

    // Cache hit.
    const auto it = executable_bundles_.find(KeyView(input_shapes));
    if (ABSL_PREDICT_TRUE(it != executable_bundles_.end())) {
      return it->second;
    }

    // Cache miss.

    std::tie(promise, future) =
        tsl::MakePromise<SharedCachedExecutableBundle>();
    if (fully_initialized_) {
      promise.Set(absl::FailedPreconditionError(
          "Got new input shapes requiring a new executable, but "
          "can't create it after full initialization. Please warm "
          "up the model with all the input shapes that are possible during "
          "serving. (See go/tf-model-warmup for more details.)"));
      return future;
    }

    executable_bundles_.emplace(Key(input_shapes), future);

    // Look up the MLIR module bundle for the given input shapes, since
    // there might be multiple MLIR modules with different input shapes as
    // the result of concretizing a polymorphic StableHLO to a set of concrete
    // StableHLOs.
    absl::StatusOr<std::unique_ptr<LoadedStableHlo::ModuleBundle>> bundle =
        loaded_stablehlo_->CloneModuleBundle(input_shapes);
    if (!bundle.ok()) {
      promise.Set(bundle.status());
      return future;
    }
    module_bundle = std::move(*bundle);
  }

  LOG(INFO) << "Cache missed. Building executable";
  if (VLOG_IS_ON(1)) {
    jsv::DumpMlirOpToFile("stablehlo_module_before_compilation",
                          module_bundle->module.get());
  }

  StablehloOptimizer::Args args;
  args.variable_indices.resize(module_bundle->input_offset);
  std::iota(args.variable_indices.begin(), args.variable_indices.end(), 0);
  if (compile_options_.has_value()) {
    args.compile_options = &*compile_options_;
  }
  absl::StatusOr<SharedCachedExecutableBundle> executable_bundle =
      CreateExecutableSynchronously(input_shapes, input_dtypes,
                                    std::move(module_bundle), args);
  if (!executable_bundle.ok()) {
    LOG(ERROR) << "Failed to create executable: "
               << executable_bundle.status().message();
  }
  promise.Set(std::move(executable_bundle));
  return future;
}

bool StableHloRunner::IsPortableExecutable() const {
  return serving_core_selector_ != nullptr &&
         assigned_devices_->num_replicas == 1 &&
         assigned_devices_->num_cores_per_replica == 1;
}

absl::StatusOr<std::vector<std::optional<xla::ifrt::ArrayRef>>>
StableHloRunner::LookupWeights(std::optional<size_t> device_id) {
  std::vector<int> device_ids;
  if (device_id.has_value()) {
    device_ids.push_back(*device_id);
  } else {
    device_ids.reserve(assigned_device_list_->size());
    for (const auto& device : assigned_device_list_->devices()) {
      device_ids.push_back(device->Id().value());
    }
  }

  const int num_weights = weights_.size();

  // Ifrt Arrays for the weights. When an element is set to std::nullopt, it
  // means the weight is unused (i.e. does not needed to be passed to the
  // StableHLO for execution).
  std::vector<std::optional<xla::ifrt::ArrayRef>> arrays(num_weights,
                                                         std::nullopt);

  // Get weight specs. The weight spec will be nullopt if the weight is unused
  // by the StableHLO.
  const std::vector<std::optional<ArgSpec>>& weights_specs =
      loaded_stablehlo_->weight_specs();
  if (weights_specs.size() != num_weights) {
    return absl::FailedPreconditionError(
        absl::StrCat("Expected ", weights_.size(), " weight specs, but got ",
                     weights_specs.size(), " weight specs."));
  }

  // Build IFRT arrays for the weights. The array will be nullopt if the weight
  // is unused.
  for (int i = 0; i < num_weights; i++) {
    if (!weights_specs[i].has_value()) {
      // Unused weights. They should never be needed when executing StableHLO.
      continue;
    }

    const ArgSpec& weight_spec = weights_specs[i].value();
    const auto& weight_name = weights_[i];
    std::optional<xla::Layout> layout = std::nullopt;
    if (weight_spec.layout.has_value()) {
      layout = *weight_spec.layout;
    }
    IfrtLoadedVariableRegistry::Key key{
        .device_ids = device_ids,
        .input_name = weight_name,
        .hlo_sharding = device_id.has_value()
                            ? xla::HloSharding::AssignDevice(*device_id)
                            : weight_spec.sharding,
        .layout = layout,
    };
    TF_ASSIGN_OR_RETURN(auto loaded_weight,
                        ifrt_loaded_variable_registry_.GetLoadedVariable(key));
    TF_ASSIGN_OR_RETURN(
        xla::ifrt::ArrayRef loaded_weight_ifrt_array,
        loaded_weight.value.Get<tsl::Future<xla::ifrt::ArrayRef>>().Await());
    // Check the shape and DType of loaded weight matches the weight spec.
    // Make sure the shape and dtype of IFRT array for the weight matches the
    // the spec of the weight.
    if (loaded_weight_ifrt_array->shape() != ToIfrtShape(weight_spec.shape)) {
      return absl::FailedPreconditionError(absl::StrCat(
          "StableHlo expected weight #", i, " (", weights_[i],
          ")"
          " to have shape: ",
          ToIfrtShape(weight_spec.shape).DebugString(), ", but got ",
          loaded_weight_ifrt_array->shape().DebugString(),
          " from the actual weight."));
    }
    if (loaded_weight_ifrt_array->dtype() != weight_spec.dtype) {
      return absl::FailedPreconditionError(absl::StrCat(
          "StableHlo expected weight #", i, " '", weights_[i],
          "' to have dtype: ", weight_spec.dtype, ", but got ",
          loaded_weight_ifrt_array->dtype(), " from the actual weight."));
    }
    TF_RETURN_IF_ERROR(
        InputShapeEqual(weight_spec.shape, loaded_weight_ifrt_array->shape()));

    arrays[i] = std::move(loaded_weight_ifrt_array);
  }
  return arrays;
}

absl::StatusOr<std::vector<HostBuffer>> StableHloRunner::Execute(
    absl::Span<const HostBuffer> inputs) {
  tsl::profiler::TraceMe trace_me([&]() {
    return tsl::profiler::TraceMeEncode("StableHloRunner::Execute", {});
  });

  VLOG(2) << "Start StableHloRunner execution";

  if (inputs.size() != loaded_stablehlo_->num_user_inputs()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected ", loaded_stablehlo_->num_user_inputs(),
                     " user inputs, but got ", inputs.size()));
  }

  // TODO: Consider moving this to `Init()`. Maybe also remove the
  // `call_once` then.
  absl::call_once(weights_lookup_once_, [&]() {
    if (IsPortableExecutable()) {
      for (auto& [device_list, resources] : device_list_to_resources_map_) {
        resources->weight_arrays =
            LookupWeights(device_list->devices()[0]->Id().value());
      }
    } else {
      device_list_to_resources_map_[assigned_device_list_]->weight_arrays =
          LookupWeights();
    }
  });

  // Lookup the executable bundle in the cache.
  std::vector<xla::ifrt::Shape> input_shapes;
  std::vector<xla::ifrt::DType> input_dtypes;
  input_shapes.reserve(inputs.size());
  for (const auto& input : inputs) {
    input_shapes.push_back(input.shape());
    input_dtypes.push_back(input.dtype());
  }
  TF_ASSIGN_OR_RETURN(
      SharedCachedExecutableBundle executable_bundle,
      LookUpOrCreateExecutable(input_shapes, input_dtypes).Await());

  // Find the resources per the device(s) to use. If the executable is portable,
  // use the core selector to reserve a device first.
  tsl::DeviceReservation device_reservation(/*device_index=*/-1,
                                            /*selector=*/nullptr);
  xla::ifrt::DeviceListRef device_list = assigned_device_list_;
  if (IsPortableExecutable()) {
    device_reservation = serving_core_selector_->ReserveDevice(
        executable_bundle->program_fingerprint_with_timestamp);
    const int device_id = device_reservation.device_index();
    TF_ASSIGN_OR_RETURN(
        xla::ifrt::Device * device,
        ifrt_client_->LookupDevice(xla::ifrt::DeviceId(device_id)));
    TF_ASSIGN_OR_RETURN(device_list, ifrt_client_->MakeDeviceList({device}));
  }
  const auto resources_it = device_list_to_resources_map_.find(device_list);
  if (resources_it == device_list_to_resources_map_.end()) {
    return absl::InternalError(
        absl::StrCat("Device list resources not initialized for device list: ",
                     device_list->DebugString()));
  }
  TF_ASSIGN_OR_RETURN(const auto& weights_array,
                      resources_it->second->weight_arrays);
  DataConverter* data_converter = resources_it->second->data_converter.get();

  // `used_input_indices` is used to track the index of used inputs in
  // stablehlo spec because some inputs may be presented in the model signature
  // but not used by StableHlo.
  std::vector<int> used_input_indices;
  for (int idx : executable_bundle->module_kept_var_idx) {
    if (idx >= weights_array.size()) {
      used_input_indices.push_back(idx - weights_array.size());
    }
  }

  std::vector<HostBuffer> reshaped_inputs;
  reshaped_inputs.reserve(inputs.size());
  // `input_idx` is used to track the index of used inputs in stablehlo spec.
  for (int i = 0; i < used_input_indices.size(); ++i) {
    int idx_in_stablehlo_spec = i + executable_bundle->input_offset;
    int original_input_idx = used_input_indices[i];
    // If the optimizer is not set, the input shape won't be changed. If the
    // input is already rank 1 or is a scalar, we can copy the input without
    // doing any reshaping.
    if (optimizer_ == nullptr ||
        inputs[original_input_idx].shape().dims().size() <= 1) {
      reshaped_inputs.push_back(std::move(inputs[original_input_idx]));
      continue;
    }
    // R1 optimization is applied, the shape expected by StableHlo is changed
    // to rank 1, so here we need to reshape the input to rank 1.
    xla::ifrt::Shape shape = ToIfrtShape(
        executable_bundle->io_spec.inputs[idx_in_stablehlo_spec].shape);
    // The original HostBuffer in `inputs` will be alive during the execution,
    // so there is no need to give the `reshaped_buffer` an owner.
    HostBuffer reshaped_buffer(inputs[original_input_idx].base(),
                               inputs[original_input_idx].dtype(), shape);
    reshaped_inputs.push_back(std::move(reshaped_buffer));
  }

  std::vector<xla::ifrt::ArrayRef> inputs_with_weights;
  inputs_with_weights.reserve(executable_bundle->module_kept_var_idx.size());

  // TODO: Consider validating that the inputs and outputs match
  // the model IO spec before converting.
  TF_ASSIGN_OR_RETURN(
      auto converted_inputs,
      data_converter->ConvertInput(reshaped_inputs, executable_bundle->io_spec,
                                   executable_bundle->input_offset));

  for (int idx : executable_bundle->module_kept_var_idx) {
    if (idx < weights_array.size()) {
      const auto& weight_array = weights_array[idx];
      if (!weight_array.has_value()) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Weight #", idx,
            " is required by the StableHLO program but was not loaded. This "
            "usually indicates the weight was incorrectly treated as unused, "
            "thus not being loaded."));
      }
      inputs_with_weights.push_back(weight_array.value());
    }
  }
  // `converted_inputs` only contains used inputs, so we just insert them
  // at the end of the list.
  inputs_with_weights.insert(inputs_with_weights.end(),
                             std::make_move_iterator(converted_inputs.begin()),
                             std::make_move_iterator(converted_inputs.end()));
  TF_ASSIGN_OR_RETURN(
      auto execution_result,
      executable_bundle->ifrt_executable->Execute(
          absl::MakeSpan(inputs_with_weights),
          /*options=*/
          {
              .launch_id = static_cast<int32_t>(launch_id_.fetch_add(1)),
              .fill_status = true,
          },
          /*devices=*/
          IsPortableExecutable()
              ? std::make_optional(data_converter->device_list())
              : std::nullopt));
  TF_RETURN_IF_ERROR(execution_result.status.Await());
  return data_converter->ConvertOutput(absl::MakeSpan(execution_result.outputs),
                                       executable_bundle->io_spec);
}

absl::Status StableHloRunner::MarkFullyInitialized() {
  absl::MutexLock lock(mutex_);
  fully_initialized_ = true;
  // Release the module as it is no longer needed after fully initialized
  // because no compilation will be allowed. A module can be large, e.g.
  // containing a large const.
  loaded_stablehlo_->ReleaseAllModules();
  return absl::OkStatus();
}

}  // namespace jsv
