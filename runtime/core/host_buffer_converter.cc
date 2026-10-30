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

#include "runtime/core/host_buffer_converter.h"  // from @xla_serving

#include <memory>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"  // from @abseil-cpp
#include "absl/base/no_destructor.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "runtime/core/value.h"  // from @xla_serving

namespace xla_serving {

HostBufferConverterRegistry& HostBufferConverterRegistry::Global() {
  static absl::NoDestructor<HostBufferConverterRegistry> registry;
  return *registry;
}

std::shared_ptr<const HostBufferConverterRegistry::ConverterList>
HostBufferConverterRegistry::Snapshot() const {
  absl::ReaderMutexLock lock(snapshot_mu_);
  return converters_;
}

void HostBufferConverterRegistry::Publish(
    std::shared_ptr<const ConverterList> converters) {
  {
    absl::MutexLock lock(snapshot_mu_);
    converters_.swap(converters);
  }
  // `converters` now holds the previous snapshot; it is released here, outside
  // `snapshot_mu_`, or later by the last reader still using it.
}

absl::Status HostBufferConverterRegistry::RegisterImpl(
    std::string name, HostBufferConverter::TypeKey type,
    HostBufferConverter::Matcher matches,
    HostBufferConverter::ErasedFn convert) {
  absl::MutexLock write_lock(write_mu_);
  std::shared_ptr<const ConverterList> current = Snapshot();
  auto existing = absl::c_find_if(
      *current, [type](const auto& entry) { return entry->type_ == type; });
  if (existing != current->end()) {
    return absl::AlreadyExistsError(absl::StrCat(
        "A HostBuffer converter for this type is already registered ('",
        (*existing)->name(), "'); refusing to register '", name, "'."));
  }
  auto next = std::make_shared<ConverterList>(*current);
  // `HostBufferConverter`'s constructor is private, so no `make_shared`.
  next->push_back(
      std::shared_ptr<const HostBufferConverter>(new HostBufferConverter(
          std::move(name), type, matches, std::move(convert))));
  Publish(std::move(next));
  return absl::OkStatus();
}

bool HostBufferConverterRegistry::UnregisterImpl(
    HostBufferConverter::TypeKey type) {
  absl::MutexLock write_lock(write_mu_);
  std::shared_ptr<const ConverterList> current = Snapshot();
  auto it = absl::c_find_if(
      *current, [type](const auto& entry) { return entry->type_ == type; });
  if (it == current->end()) return false;
  auto next = std::make_shared<ConverterList>(*current);
  next->erase(next->begin() + (it - current->begin()));
  Publish(std::move(next));
  return true;
}

std::shared_ptr<const HostBufferConverter> HostBufferConverterRegistry::Find(
    const Value& value) const {
  if (!value.HasValue()) return nullptr;
  const std::shared_ptr<const ConverterList> converters = Snapshot();
  // At most one converter per type, so the order does not matter.
  for (const auto& converter : *converters) {
    if (converter->matches_(value)) return converter;
  }
  return nullptr;
}

std::shared_ptr<const HostBufferConverter> FindHostBufferConverter(
    const Value& value) {
  return HostBufferConverterRegistry::Global().Find(value);
}

}  // namespace xla_serving
