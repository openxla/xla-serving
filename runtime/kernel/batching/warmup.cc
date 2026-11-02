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

#include "runtime/kernel/batching/warmup.h"  // from @xla_serving

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/time/time.h"  // from @abseil-cpp
#include "xla/tsl/lib/monitoring/sampler.h"  // from @xla

namespace xla_serving {
namespace {

auto* model_warmup_latency = tsl::monitoring::Sampler<3>::New(
    {"/orbax/serving/model_warmup_latency",
     "Distribution of wall time (in microseconds) for warming up a model, "
     "broken down by model name, version and status code.",
     "model_name", "model_version", "status"},
    // 100 ms to ~83 min.
    tsl::monitoring::Buckets::Exponential(100000, 1.6, 24));

}  // namespace

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

void RecordModelWarmupLatency(absl::string_view model_name,
                              int64_t model_version, absl::Duration latency,
                              const absl::Status& status) {
  model_warmup_latency
      ->GetCell(model_name, absl::StrCat(model_version),
                absl::StatusCodeToString(status.code()))
      ->Add(absl::ToDoubleMicroseconds(latency));
}

}  // namespace xla_serving
