#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_BATCHING_WARMUP_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_BATCHING_WARMUP_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp

namespace jsv {

// Global registry for model's warm-up states. Before a model executes warm-up
// requests, it is registered here so that the runtime can distinguish demand
// requests vs. warm-up requests and apply warm-up specific optimizations.
class WarmupStateRegistry {
 public:
  struct Key {
    std::string name;
    int64_t version;

    Key(std::string name, int64_t version)
        : name(std::move(name)), version(version) {}

    template <typename H>
    friend H AbslHashValue(H state, const Key& key) {
      return H::combine(std::move(state), key.name, key.version);
    }

    friend bool operator==(const Key& x, const Key& y) {
      return x.name == y.name && x.version == y.version;
    }
  };
  // Data stored per key.
  struct PerModelData {
    // If true, supported batch ops will execute the model on dummy batches
    // for all `allowed_batch_sizes` of that batch op. This removes the
    // need to issue separate warmup requests for each batch size.
    bool warmup_all_batch_sizes = false;
  };

  // RAII handle for registered models.
  class Handle {
   public:
    Handle() = default;

    Handle(const Handle& other) = delete;
    Handle& operator=(const Handle& other) = delete;
    Handle(Handle&& other)
        : key_(std::move(other.key_)), registry_(other.registry_) {
      other.key_.reset();
    }
    Handle& operator=(Handle&& other) {
      if (key_.has_value()) {
        Release();
      }

      key_ = std::move(other.key_);
      other.key_.reset();
      registry_ = other.registry_;
      return *this;
    }

    ~Handle() { Release(); }

    void Release();

   private:
    friend class WarmupStateRegistry;

    // Can only be constructed by `WarmupStateRegistry::Register()`.
    Handle(const Key& key, WarmupStateRegistry* registry)
        : key_(key), registry_(registry) {
      DCHECK(registry_);
    }

    std::optional<Key> key_;
    WarmupStateRegistry* registry_ = nullptr;
  };

  // Registers the given model to be in a warm-up state and associates the given
  // metadata with the model. Returns an RAII handle that unregisters the model
  // at its destruction.
  absl::StatusOr<Handle> Register(const Key& model_key,
                                  std::unique_ptr<PerModelData> per_model_data);

  // Return model data. A nullptr indicates the key was not present.
  const PerModelData* Lookup(const Key& model_key);

 private:
  friend class Handle;

  void Unregister(const Key& model_key);

  absl::Mutex mu_;
  // Map of model names/versions to miscellaneous data.
  absl::flat_hash_map<Key, std::unique_ptr<PerModelData>> states_
      ABSL_GUARDED_BY(&mu_);
};

WarmupStateRegistry& GetGlobalWarmupStateRegistry();

// Utility function that returns whether or not to warmup all batch sizes,
// based on the state of WarmupStateRegistry.
bool ShouldWarmupAllBatchSizes(absl::string_view model_name,
                               int64_t model_version);

}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_BATCHING_WARMUP_H_
