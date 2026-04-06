#include "runtime/data_loader/default_data_loader.h"  // from @neptune

#include <cstddef>
#include <iterator>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @abseil-cpp
#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/base/optimization.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/container/flat_hash_set.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "protos/manifest.pb.h"  // from @neptune
#include "runtime/core/constants.h"  // from @neptune
#include "runtime/core/ifrt_device_utils.h"  // from @neptune
#include "runtime/core/ifrt_loaded_variable_registry.h"  // from @neptune
#include "runtime/core/stablehlo_utils.h"  // from @neptune
#include "runtime/core/value.h"  // from @neptune
#include "runtime/data_loader/data_loader.h"  // from @neptune
#include "runtime/data_loader/ocp_loader/checkpoint_loader.h"  // from @neptune
#include "runtime/data_loader/ocp_loader/metadata_util.h"  // from @neptune
#include "runtime/interpreter/manifest_utils.h"  // from @neptune
#include "xla/hlo/ir/hlo_sharding.h"  // from @xla
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/tsl/concurrency/future.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla

namespace neptune {
namespace {

// Result of the `Load` method.
struct LoadedWeightArrays {
  // A list of {weight_name, weight_ifrt_array} for loaded weights.
  // Unused weights are not loaded and not included in this list.
  std::vector<std::pair<std::string, xla::ifrt::ArrayRef>> named_weight_arrays;
};

absl::Status ValidateLoadInputs(
    const neptune_model_manifest::ExternalValue& value_to_load,
    std::optional<absl::string_view> model_path,
    std::optional<const neptune_model_manifest::Manifest*> manifest,
    std::optional<absl::Span<const neptune_model_manifest::Function* const>>
        stablehlo_funcs) {
  if (!value_to_load.has_data()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Checkpoint value has no `data`: ", value_to_load));
  }
  TF_RETURN_IF_ERROR(CheckMimeAndDataType(
      value_to_load.data(), {kOrbaxCheckpointMimeType},
      neptune_model_manifest::UnstructuredData::kFileSystemLocation,
      "Checkpoint value"));
  if (!model_path.has_value()) {
    return absl::InvalidArgumentError(
        "`model_path` is required for Orbax checkpoint loading.");
  }
  if (!manifest.has_value()) {
    return absl::InvalidArgumentError(
        "`manifest` is required for Orbax checkpoint loading.");
  }
  if (!stablehlo_funcs.has_value()) {
    return absl::InvalidArgumentError(
        "`stablehlo_funcs` is required for Orbax checkpoint loading.");
  }
  if (stablehlo_funcs.value().empty()) {
    return absl::InvalidArgumentError(
        "`stablehlo_funcs` should not be empty for Orbax checkpoint loading.");
  }
  return absl::OkStatus();
}

struct DeviceData {
  xla::ifrt::DeviceListRef device_list;
  std::vector<int> device_ids;
};

absl::StatusOr<DeviceData> GetDeviceData(
    const xla::ifrt::Client& ifrt_client,
    const AssignedDevices* absl_nullable assigned_devices) {
  if (assigned_devices == nullptr) {
    return absl::InvalidArgumentError(
        "`assigned_devices` should be non-null for `DefaultDataLoader` "
        "loading");
  }
  TF_ASSIGN_OR_RETURN(
      const std::vector<xla::ifrt::Device*> devices,
      GetAssignedIfrtDevices(ifrt_client, assigned_devices->num_replicas,
                             assigned_devices->num_cores_per_replica,
                             assigned_devices->device_assignment));
  DeviceData device_data;
  TF_ASSIGN_OR_RETURN(device_data.device_list,
                      ifrt_client.MakeDeviceList(devices));
  device_data.device_ids.reserve(device_data.device_list->size());
  for (const auto& device : device_data.device_list->devices()) {
    device_data.device_ids.push_back(device->Id().value());
  }
  return device_data;
}

struct LoadOrbaxCheckpointTask {
  std::string checkpoint_path;
  std::vector<std::string> weight_names;
  std::vector<std::optional<ArgSpec>> weight_specs;
  xla::ifrt::DeviceListRef device_list;
  xla::ifrt::Client* client_ptr;
  absl::flat_hash_map<std::string,
                      std::shared_ptr<tsl::Promise<xla::ifrt::ArrayRef>>>
      weight_array_promises;

  void operator()() {
    absl::Status status = [&]() -> absl::Status {
      const int num_weights = weight_names.size();

      std::vector<std::string> used_weight_names;
      used_weight_names.reserve(num_weights);
      std::vector<neptune::ArraySpec> used_weight_array_specs;
      used_weight_array_specs.reserve(num_weights);

      for (int i = 0; i < num_weights; ++i) {
        absl::string_view weight_name = weight_names[i];
        const std::optional<ArgSpec>& weight_spec = weight_specs[i];
        if (ABSL_PREDICT_FALSE(!weight_spec.has_value())) {
          continue;
        }

        // We only load weights for which promises were pre-registered.
        if (!weight_array_promises.contains(weight_name)) {
          continue;
        }
        used_weight_names.push_back(std::string(weight_name));
        used_weight_array_specs.push_back(neptune::ArraySpec{
            .dtype = weight_spec->dtype,
            .shape = ToIfrtShape(weight_spec->shape),
            .sharding = weight_spec->sharding.ToProto(),
            .layout = weight_spec->layout,
        });
      }

      TF_ASSIGN_OR_RETURN(
          auto named_weight_arrays,
          neptune::LoadCheckpointFromOrbax(
              checkpoint_path, used_weight_names, used_weight_array_specs,
              device_list, *client_ptr));

      // Fulfill the promises with the loaded arrays.
      for (const auto& [weight_name, weight_array] : named_weight_arrays) {
        auto it = weight_array_promises.find(weight_name);
        if (it != weight_array_promises.end()) {
          it->second->Set(weight_array);
        } else {
          return absl::InternalError(absl::StrCat(
              "Loaded weight '", weight_name, "' but no promise was found."));
        }
      }
      return absl::OkStatus();
    }();

    if (!status.ok()) {
      // If loading failed, set all pending promises to the error status.
      for (auto& [weight_name, promise] : weight_array_promises) {
        promise->Set(status);
      }
    }
  }
};

// Loads weights from an Orbax checkpoint.
// Args:
//  - checkpoint_path: the path to the checkpoint.
//  - weight_names: the names of the weights to load. Must be the same length
//    as `weight_specs`.
//  - weight_specs: the specs of the weights to load. A weight spec will be
//    std::nullopt if the weight is unused, so loading will be skipped. Must be
//    the same length as `weight_names`.
//  - device_list: the device list to load the weights to.
//  - client: the IFRT client to use for create IFRT arrays for loaded weights.
absl::Status LoadOrbaxCheckpointWrapper(
    absl::string_view checkpoint_path,
    absl::Span<const std::string> weight_names,
    absl::Span<const std::optional<ArgSpec>> weight_specs,
    const xla::ifrt::DeviceListRef device_list, xla::ifrt::Client& client,
    absl::flat_hash_map<std::string,
                        std::shared_ptr<tsl::Promise<xla::ifrt::ArrayRef>>>
        weight_array_promises,
    tsl::thread::ThreadPool* absl_nullable thread_pool) {
  if (weight_names.size() != weight_specs.size()) {
    // Currently unreachable; kept for clarity and defensive programming.
    absl::Status error = absl::InvalidArgumentError(
        absl::StrCat("`weight_names` and `weight_specs` must have the same "
                     "size as the total number of weights, but got ",
                     weight_names.size(), " and ", weight_specs.size()));
    // Propagate errors to promises so consumers don't block forever.
    for (auto& [name, promise] : weight_array_promises) {
      promise->Set(error);
    }
    return error;
  }
  LoadOrbaxCheckpointTask task{
      std::string(checkpoint_path),
      std::vector<std::string>(weight_names.begin(), weight_names.end()),
      std::vector<std::optional<ArgSpec>>(weight_specs.begin(),
                                          weight_specs.end()),
      device_list,
      &client,
      std::move(weight_array_promises)};

  if (thread_pool != nullptr) {
    thread_pool->Schedule(std::move(task));
  } else {
    task();
  }

  // This future is no longer used to return the loaded arrays,
  // as promises are fulfilled directly in the task.
  // We return a dummy future here.
  return absl::OkStatus();
}

absl::Status AllDataNamesExistInCheckpoint(
    absl::Span<const std::string> data_names,
    absl::Span<const std::string> all_weight_names) {
  std::vector<std::string> missing_names;
  absl::c_set_difference(
      std::set<std::string>(data_names.begin(), data_names.end()),
      std::set<std::string>(all_weight_names.begin(), all_weight_names.end()),
      std::back_inserter(missing_names));
  if (!missing_names.empty()) {
    return absl::FailedPreconditionError(
        absl::StrCat("The following data names specified in `Function` are not "
                     "found in the checkpoint: ",
                     absl::StrJoin(missing_names, ", ")));
  }
  return absl::OkStatus();
}

// Returns the data names from `Function`(s).
// When there are multiple functions, verify that they should have the same
// set of data names.
absl::StatusOr<std::vector<std::string>> GetDataNamesFromFunction(
    absl::Span<const neptune_model_manifest::Function* const> stablehlo_funcs) {
  if (stablehlo_funcs.empty()) {
    return absl::InvalidArgumentError(
        "`stablehlo_funcs` should not be empty for Orbax checkpoint "
        "loading.");
  }
  // TODO: Maybe use the signature to find the data names (and
  // replace `data_names` with `additional_data_names`).
  const auto& func = stablehlo_funcs[0];
  std::vector<std::string> data_names(func->data_names().begin(),
                                      func->data_names().end());
  absl::flat_hash_set<std::string> data_names_set(data_names.begin(),
                                                  data_names.end());
  for (int i = 1; i < stablehlo_funcs.size(); ++i) {
    const auto& cur_func = stablehlo_funcs[i];
    if (cur_func->data_names().size() != data_names.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "All functions should have the same number of data names, but diffs "
          "for function at index 0 vs index ",
          i));
    }
    for (const auto& data_name : cur_func->data_names()) {
      if (!data_names_set.contains(data_name)) {
        return absl::InvalidArgumentError(absl::StrCat(
            "All functions should have the same data names, but diffs for  "
            "function at index 0 vs index ",
            i, ", data name: ", data_name,
            " is not found in function index 0."));
      }
    }
  }

  return data_names;
}

// Returns the weight names to load. The returned weight names could be empty
// indicating there is no weight to load.
absl::StatusOr<std::vector<std::string>> GetWeightNames(
    absl::string_view checkpoint_path,
    std::optional<const neptune_model_manifest::Manifest*> manifest,
    std::optional<absl::Span<const neptune_model_manifest::Function* const>>
        stablehlo_funcs) {
  TF_ASSIGN_OR_RETURN(
      std::vector<std::string> weight_names_from_checkpoint,
      neptune::GetAllWeightNamesFromCheckpoint(checkpoint_path));

  TF_ASSIGN_OR_RETURN(std::vector<std::string> data_names,
                      GetDataNamesFromFunction(stablehlo_funcs.value()));
  if (data_names.empty()) {
    // TODO: Remove this fallback once the data_names field is
    // correctly populated in the manifest.
    LOG(WARNING)
        << "`data_names` from manifest is empty. Falling back to use all "
           "weights from checkpoint.";
    if (weight_names_from_checkpoint.empty()) {
      LOG(WARNING) << "No weights found in the manifest or checkpoint. No "
                      "checkpoint loading is required.";
      return std::vector<std::string>();
    }
    data_names = std::move(weight_names_from_checkpoint);
  } else {
    TF_RETURN_IF_ERROR(AllDataNamesExistInCheckpoint(
        data_names, weight_names_from_checkpoint));
  }
  return data_names;
}

// Returns the weight specs from `Function`(s). When a weight spec in the
// returned vector is std::nullopt, it means the weight is unused, meaning it
// does not needed to be passed to the StableHLO function when executing it.
// Args
//  - num_weights: the number of weights to load irrespective of whether they
//    are used or not (i.e., passed to StableHLO during the execution)
//  - stablehlo_funcs: the function(s) to get the weight specs from. When
//    there are multiple functions (i.e. concretized StableHLOs for a given
//    polymorphic StableHLO), they should all have the same spec for used
//    weights.
absl::StatusOr<std::vector<std::optional<ArgSpec>>> GetWeightSpecsFromFunction(
    int num_weights,
    absl::Span<const neptune_model_manifest::Function* const> stablehlo_funcs) {
  std::vector<std::optional<ArgSpec>> weight_specs(num_weights, std::nullopt);
  for (int i = 0; i < stablehlo_funcs.size(); ++i) {
    const auto& func = *stablehlo_funcs[i];

    TF_ASSIGN_OR_RETURN(std::vector<ArgSpec> input_specs, GetInputSpec(func));

    if (input_specs.size() < func.data_names().size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Number of weights to load (", func.data_names().size(),
                       ") should not exceed total number of model inputs (",
                       input_specs.size(), ")."));
    }

    // TODO: Move these checks to a util function.
    if (!func.has_body()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Function ", i, " has no body."));
    }
    if (!func.body().has_stable_hlo_body()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Function ", i, " has no stable_hlo_body."));
    }
    if (func.body().stable_hlo_body().module_kept_var_idx().empty()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Function ", i, " has no or empty module_kept_var_idx."));
    }
    for (int kept_var_idx :
         func.body().stable_hlo_body().module_kept_var_idx()) {
      if (kept_var_idx >= num_weights) {
        break;
      }
      const auto& weight_spec = weight_specs[kept_var_idx];
      // TODO: Move the following to a util function.
      if (weight_spec.has_value()) {
        // Verify that the weight spec is the same across all functions.
        if (weight_spec.value() != input_specs[kept_var_idx]) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Weight spec mismatch at index ", kept_var_idx, " : ",
              weight_spec.value().DebugString(), " vs ",
              input_specs[kept_var_idx].DebugString(), " ",
              "This typically indicates inconsistent function signatures "
              "across concretized polymorphic functions"));
        }
      } else {
        weight_specs[kept_var_idx] = std::move(input_specs[kept_var_idx]);
      }
    }
  }
  return weight_specs;
}

absl::Status RegisterAndLoadCheckpoints(
    const DeviceData& device_data, absl::Span<const std::string> weight_names,
    absl::Span<const std::optional<ArgSpec>> weight_specs,
    IfrtLoadedVariableRegistry* ifrt_loaded_variable_registry,
    absl::string_view checkpoint_path, xla::ifrt::Client* ifrt_client,
    tsl::thread::ThreadPool* checkpoint_loader_thread_pool) {
  // Pre-register promises for all used weights.
  absl::flat_hash_map<std::string,
                      std::shared_ptr<tsl::Promise<xla::ifrt::ArrayRef>>>
      weight_array_promises;
  std::vector<std::string> not_loaded_weight_names;
  not_loaded_weight_names.reserve(weight_names.size());
  std::vector<std::optional<ArgSpec>> not_loaded_weight_array_specs;
  not_loaded_weight_array_specs.reserve(weight_names.size());
  for (int i = 0; i < weight_names.size(); ++i) {
    const auto& weight_name = weight_names[i];
    const auto& weight_spec = weight_specs[i];
    if (weight_spec.has_value()) {
      IfrtLoadedVariableRegistry::Key key = {
          .device_ids = device_data.device_ids,
          .input_name = weight_name,
          .hlo_sharding = weight_spec->sharding,
          .layout = weight_spec->layout,
      };
      auto [array_promise, array_future] =
          tsl::MakePromise<xla::ifrt::ArrayRef>();

      absl::Status status =
          ifrt_loaded_variable_registry->TryRegisterLoadedVariable(
              key,
              [array_future = std::move(array_future)]()
                  -> absl::StatusOr<
                      IfrtLoadedVariableRegistry::LoadedVariable> {
                return IfrtLoadedVariableRegistry::LoadedVariable{
                    .value = Value(std::move(array_future))};
              });
      if (absl::IsAlreadyExists(status)) {
        LOG(INFO) << "Skipping loading duplicated Weight: " << weight_name
                  << " already loaded.";
        continue;
      }

      if (status.ok()) {
        weight_array_promises[weight_name] =
            std::move(array_promise).ToShared();
        not_loaded_weight_names.push_back(weight_name);
        not_loaded_weight_array_specs.push_back(*weight_spec);
      } else {
        // Currently unreachable; kept for clarity and defensive programming.
        LOG(ERROR) << "Failed to register loaded variable for '" << weight_name
                   << "': " << status;
        // Propagate errors to promises so consumers don't block forever.
        array_promise.Set(status);
      }
    }
  }

  // Schedule the Orbax checkpoint loading task. Errors within
  // `LoadOrbaxCheckpointWrapper` or the scheduled task will cause the promises
  // in `weight_array_promises` to be set with the error status.
  return LoadOrbaxCheckpointWrapper(
      checkpoint_path, not_loaded_weight_names, not_loaded_weight_array_specs,
      device_data.device_list, *ifrt_client, std::move(weight_array_promises),
      checkpoint_loader_thread_pool);
}

}  // namespace

// TODO: `value_to_load` is actually the path to the checkpoint
// inside the model directory. Should update all relevant names.
absl::StatusOr<DataLoader::LoadResult> DefaultDataLoader::Load(
    const neptune_model_manifest::ExternalValue& value_to_load,
    std::optional<absl::string_view> model_path,
    const AssignedDevices* absl_nullable assigned_devices,
    std::optional<const neptune_model_manifest::Manifest*> manifest,
    std::optional<absl::Span<const neptune_model_manifest::Function* const>>
        stablehlo_funcs) {
  TF_RETURN_IF_ERROR(
      ValidateLoadInputs(value_to_load, model_path, manifest, stablehlo_funcs));

  TF_ASSIGN_OR_RETURN(
      const std::string checkpoint_path,
      GetAbsoluteFileSystemPath(model_path.value(),
                                value_to_load.data().file_system_location()));

  // Get the weight names, irrespective of they are used or not.
  TF_ASSIGN_OR_RETURN(
      std::vector<std::string> weight_names,
      GetWeightNames(checkpoint_path, manifest, stablehlo_funcs));
  if (weight_names.empty()) {
    // No weight names found, indicating nothing to load.
    return DataLoader::LoadResult{};
  }

  // Get the weight specs from the function(s). The weight spec will be
  // std::nullopt if the weight is unused.
  TF_ASSIGN_OR_RETURN(
      std::vector<std::optional<ArgSpec>> weight_specs,
      GetWeightSpecsFromFunction(weight_names.size(), stablehlo_funcs.value()));

  // Register and load the checkpoints.
  if (assigned_devices != nullptr) {
    TF_ASSIGN_OR_RETURN(const DeviceData device_data,
                        GetDeviceData(*ifrt_client_, assigned_devices));
    TF_RETURN_IF_ERROR(RegisterAndLoadCheckpoints(
        device_data, weight_names, weight_specs, ifrt_loaded_variable_registry_,
        checkpoint_path, ifrt_client_.get(), checkpoint_loader_thread_pool_));
  } else {
    // If assigned devices are not provided, we will load the checkpoints to
    // all devices.
    // TODO: This path is not used yet, but will soon be used in
    // cl/816939381.
    for (size_t i = 0; i < ifrt_client_->addressable_devices().size(); ++i) {
      TF_ASSIGN_OR_RETURN(xla::ifrt::Device * device,
                          ifrt_client_->LookupDevice(xla::ifrt::DeviceId(i)));
      TF_ASSIGN_OR_RETURN(xla::ifrt::DeviceListRef device_list,
                          ifrt_client_->MakeDeviceList({device}));
      const DeviceData device_data = {.device_list = device_list,
                                      .device_ids = {static_cast<int>(i)}};
      for (auto& weight_spec : weight_specs) {
        if (weight_spec.has_value()) {
          weight_spec->sharding = xla::HloSharding::SingleDevice(i);
        }
      }
      TF_RETURN_IF_ERROR(RegisterAndLoadCheckpoints(
          device_data, weight_names, weight_specs,
          ifrt_loaded_variable_registry_, checkpoint_path, ifrt_client_.get(),
          checkpoint_loader_thread_pool_));
    }
  }

  return DataLoader::LoadResult{.data_names = std::move(weight_names)};
}

absl::StatusOr<std::unique_ptr<DefaultDataLoader>> DefaultDataLoader::Create(
    CreateOptions create_options) {
  if (!create_options.ifrt_client.has_value()) {
    return absl::InvalidArgumentError(
        "ifrt_client is required for DefaultDataLoader.");
  }
  return std::unique_ptr<DefaultDataLoader>(
      new DefaultDataLoader(std::move(create_options)));
}

DefaultDataLoader::DefaultDataLoader(CreateOptions create_options)
    : DataLoader(create_options),
      ifrt_client_(std::move(create_options.ifrt_client.value())),
      checkpoint_loader_thread_pool_(
          create_options.checkpoint_loader_thread_pool) {}

}  // namespace neptune
