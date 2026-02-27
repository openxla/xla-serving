#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_KERNEL_BUILDER_CONTEXT_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_KERNEL_BUILDER_CONTEXT_H_

#include <string>

#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/neptune/runtime/core/value.h"

namespace jsv {

class KernelBuilderContext {
 public:
  KernelBuilderContext() = default;
  ~KernelBuilderContext() = default;

  template <typename T>
  absl::StatusOr<T> GetConfig(absl::string_view name) const {
    if (auto it = configs_.find(name); it != configs_.end()) {
      if (!it->second.IsType<T>()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Config ", name, " is not of type ", typeid(T).name()));
      }
      return it->second.template Get<T>();
    }
    return absl::NotFoundError(absl::StrCat("Config ", name, " not found."));
  }

  template <typename T>
  absl::Status SetConfig(absl::string_view name, T value) {
    auto& placeholder = configs_[name];
    if (placeholder.HasValue()) {
      return absl::AlreadyExistsError(
          absl::StrCat("Config ", name, " already set."));
    }
    placeholder = value;
    return absl::OkStatus();
  }

 private:
  absl::flat_hash_map<std::string, Value> configs_;
};

}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_KERNEL_BUILDER_CONTEXT_H_
