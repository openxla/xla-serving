#include "third_party/neptune/runtime/kernel/batching/warmup.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "third_party/absl/log/check.h"
#include "third_party/absl/log/log.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/synchronization/mutex.h"

namespace jsv {

void WarmupStateRegistry::Handle::Release() {
  if (!key_.has_value()) {
    return;
  }

  DCHECK(registry_);
  registry_->Unregister(*key_);
}

absl::StatusOr<WarmupStateRegistry::Handle> WarmupStateRegistry::Register(
    const Key& model_key, std::unique_ptr<PerModelData> per_model_data) {
  absl::MutexLock l(mu_);
  VLOG(1) << "Registering model " << model_key.name << ":" << model_key.version
          << " to warm-up registry";
  if (!states_.insert(std::pair(model_key, std::move(per_model_data))).second) {
    return absl::AlreadyExistsError(
        absl::StrCat("Model ", model_key.name, ":", model_key.version,
                     " already exists in the warm-up registry"));
  }
  return Handle(model_key, this);
}

void WarmupStateRegistry::Unregister(const Key& model_key) {
  absl::MutexLock l(mu_);

  VLOG(1) << "Unregistering model " << model_key.name << ":"
          << model_key.version << " from warm-up registry";
  states_.erase(model_key);
}

const WarmupStateRegistry::PerModelData* WarmupStateRegistry::Lookup(
    const Key& model_key) {
  absl::ReaderMutexLock l(mu_);
  return states_.contains(model_key) ? states_[model_key].get() : nullptr;
}

WarmupStateRegistry& GetGlobalWarmupStateRegistry() {
  static auto* const registry = new WarmupStateRegistry;
  return *registry;
}

bool ShouldWarmupAllBatchSizes(absl::string_view model_name,
                               int64_t model_version) {
  if (model_name.empty()) return false;
  const WarmupStateRegistry::Key key{std::string(model_name), model_version};
  const auto* per_model_data = GetGlobalWarmupStateRegistry().Lookup(key);
  return per_model_data && per_model_data->warmup_all_batch_sizes;
}

}  // namespace jsv
