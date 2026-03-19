#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_IFRT_LOADED_VARIABLE_REGISTRY_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_IFRT_LOADED_VARIABLE_REGISTRY_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/functional/any_invocable.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "runtime/core/value.h"  // from @neptune
#include "xla/hlo/ir/hlo_sharding.h"  // from @xla
#include "xla/layout.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla

namespace neptune {

// This class is thread safe.
class IfrtLoadedVariableRegistry {
 public:
  // The key is per variable tensor per device assignment. For single -device
  // program, variables can be loaded on multiple devices with core selection.
  // For SPMD program, we currently assume all devices will be used, so we use
  // vector to make it compatible with SPMD.
  struct Key {
    // We use a vector to make it compatible with SPMD because the order of the
    // devices used for sharding must match the order of the devices used for
    // xla compilation.
    std::vector<int> device_ids;
    std::string input_name;
    xla::HloSharding hlo_sharding;
    std::optional<xla::Layout> layout;
    template <typename H>
    friend H AbslHashValue(H h, const Key& key) {
      h = H::combine(std::move(h), key.input_name, key.device_ids,
                     key.hlo_sharding, key.layout);
      return h;
    }

    friend bool operator==(const Key& x, const Key& y) {
      return x.input_name == y.input_name && x.device_ids == y.device_ids &&
             x.hlo_sharding == y.hlo_sharding && x.layout == y.layout;
    }

    std::string ToString() const {
      std::string layout_str = layout.has_value() ? layout->ToString() : "null";
      return absl::StrCat(input_name, ":", absl::StrJoin(device_ids, ","), ":",
                          hlo_sharding.ToString(), ":", layout_str);
    }
  };

  struct LoadedVariable {
    Value value;
  };
  using LoadedVariableConstructor =
      absl::AnyInvocable<absl::StatusOr<LoadedVariable>() const>;

  // Tries to register a loaded variable with the given name.
  // Returns an error if the named array does not already exists and
  // loaded_variable_constructor failed to create an array. Note that it returns
  // OK if the named array already exists.
  // loaded_variable_constructor is invoked in the caller thread.
  absl::Status TryRegisterLoadedVariable(
      const Key& key, LoadedVariableConstructor&& loaded_variable_constructor)
      ABSL_LOCKS_EXCLUDED(mutex_);

  absl::StatusOr<LoadedVariable> GetLoadedVariable(const Key& key) const
      ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  friend class IfrtLoadedVariableRegistryTestPeer;

  mutable absl::Mutex mutex_;
  absl::flat_hash_map<Key, LoadedVariable> loaded_variable_map_
      ABSL_GUARDED_BY(mutex_);
};

class IfrtLoadedVariableRegistryTestPeer {
 public:
  explicit IfrtLoadedVariableRegistryTestPeer(
      IfrtLoadedVariableRegistry* absl_nonnull registry)
      : registry_(registry) {}

  absl::flat_hash_map<IfrtLoadedVariableRegistry::Key,
                      IfrtLoadedVariableRegistry::LoadedVariable>
  GetLoadedVariableMap() {
    absl::MutexLock lock(registry_->mutex_);
    return registry_->loaded_variable_map_;
  }

 private:
  IfrtLoadedVariableRegistry* absl_nonnull registry_;
};

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_IFRT_LOADED_VARIABLE_REGISTRY_H_
