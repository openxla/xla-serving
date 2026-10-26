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
#include <optional>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"  // from @abseil-cpp
#include "absl/base/no_destructor.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/value.h"  // from @xla_serving

namespace xla_serving {

HostBufferConverterRegistry& HostBufferConverterRegistry::Global() {
  static absl::NoDestructor<HostBufferConverterRegistry> registry;
  return *registry;
}

absl::Status HostBufferConverterRegistry::RegisterImpl(
    std::string name, HostBufferConverter::TypeKey type,
    HostBufferConverter::Matcher matches,
    HostBufferConverter::ErasedFn convert) {
  absl::MutexLock lock(mu_);
  auto existing = absl::c_find_if(
      converters_, [type](const auto& entry) { return entry->type_ == type; });
  if (existing != converters_.end()) {
    return absl::AlreadyExistsError(absl::StrCat(
        "A HostBuffer converter for this type is already registered ('",
        (*existing)->name(), "'); refusing to register '", name, "'."));
  }
  // `HostBufferConverter`'s constructor is private, so no `make_shared`.
  converters_.push_back(
      std::shared_ptr<const HostBufferConverter>(new HostBufferConverter(
          std::move(name), type, matches, std::move(convert))));
  return absl::OkStatus();
}

bool HostBufferConverterRegistry::UnregisterImpl(
    HostBufferConverter::TypeKey type) {
  absl::MutexLock lock(mu_);
  auto it = absl::c_find_if(
      converters_, [type](const auto& entry) { return entry->type_ == type; });
  if (it == converters_.end()) return false;
  converters_.erase(it);
  return true;
}

std::shared_ptr<const HostBufferConverter> HostBufferConverterRegistry::Find(
    const Value& value) const {
  if (!value.HasValue()) return nullptr;
  absl::MutexLock lock(mu_);
  // At most one converter per type, so the order does not matter.
  for (const auto& converter : converters_) {
    if (converter->matches_(value)) return converter;
  }
  return nullptr;
}

absl::StatusOr<std::optional<HostBuffer>> HostBufferConverterRegistry::Convert(
    const Value& value) const {
  std::shared_ptr<const HostBufferConverter> converter = Find(value);
  if (converter == nullptr) return std::nullopt;
  absl::StatusOr<HostBuffer> host_buffer = converter->Convert(value);
  if (!host_buffer.ok()) {
    return absl::Status(
        host_buffer.status().code(),
        absl::StrCat("HostBuffer converter '", converter->name(),
                     "' failed: ", host_buffer.status().message()));
  }
  return std::optional<HostBuffer>(*std::move(host_buffer));
}

std::shared_ptr<const HostBufferConverter> FindHostBufferConverter(
    const Value& value) {
  return HostBufferConverterRegistry::Global().Find(value);
}

absl::StatusOr<std::optional<HostBuffer>> ConvertToHostBuffer(
    const Value& value) {
  return HostBufferConverterRegistry::Global().Convert(value);
}

}  // namespace xla_serving
