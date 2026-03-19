#include "runtime/core/ifrt_loaded_variable_registry.h"  // from @neptune

#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/substitute.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "xla/tsl/platform/statusor.h"  // from @xla

namespace neptune {

absl::Status IfrtLoadedVariableRegistry::TryRegisterLoadedVariable(
    const Key& key, LoadedVariableConstructor&& loaded_variable_constructor) {
  VLOG(4) << "Register variable " << key.input_name << " with sharding "
          << key.hlo_sharding << " on " << key.device_ids.size() << " devices.";

  absl::MutexLock lock(mutex_);
  TF_ASSIGN_OR_RETURN(auto loaded_variable, loaded_variable_constructor());
  const auto [unused_it, inserted] =
      loaded_variable_map_.try_emplace(key, std::move(loaded_variable));
  if (!inserted) {
    return absl::AlreadyExistsError(
        absl::StrCat("Variable '", key.input_name, "' already registered."));
  }
  return absl::OkStatus();
}

absl::StatusOr<IfrtLoadedVariableRegistry::LoadedVariable>
IfrtLoadedVariableRegistry::GetLoadedVariable(const Key& key) const {
  VLOG(4) << "Look for variable " << key.input_name << " with sharding "
          << key.hlo_sharding << " on " << key.device_ids.size() << " devices.";

  absl::MutexLock lock(mutex_);
  auto it = loaded_variable_map_.find(key);
  if (it == loaded_variable_map_.end()) {
    std::string key_with_same_name;
    for (const auto& [cur_key, _] : loaded_variable_map_) {
      if (key.input_name != cur_key.input_name) {
        continue;
      }
      absl::StrAppend(&key_with_same_name, cur_key.ToString(), ",");
    }
    return absl::NotFoundError(absl::Substitute(
        "No loaded variable found for $0, loaded variables with same names "
        "are: $1",
        key.ToString(), key_with_same_name));
  }
  return it->second;
}

}  // namespace neptune
