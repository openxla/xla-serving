#include "runtime/core/flat_signature.h"  // from @neptune

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_format.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "runtime/core/stablehlo_utils.h"  // from @neptune
#include "runtime/core/value.h"  // from @neptune
#include "third_party/py/orbax/experimental/model/core/protos/manifest.pb.h"
#include "third_party/py/orbax/experimental/model/core/protos/type.pb.h"
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla

namespace neptune {

std::string MetaArgSpec::DebugString() const {
  return absl::StrFormat("MetaArgSpec: %s, name: %s, has_default_value: %v, ",
                         spec.DebugString(), name.value_or("null"),
                         default_value.HasValue());
}

bool MetaArgSpec::EqualExceptDefaultValue(const MetaArgSpec& rhs) const {
  // It is inconvenient to compare the default value, so we only compare the
  // existence of the default value.
  return spec == rhs.spec && name == rhs.name &&
         (default_value.HasValue() == rhs.default_value.HasValue());
}

std::string FlatSignature::DebugString() const {
  std::string debug_string = "specs: [\n";
  for (int i = 0; i < specs_.size(); ++i) {
    absl::StrAppend(&debug_string, "#", i, ": ", specs_[i].DebugString(),
                    ",\n");
  }
  absl::StrAppend(&debug_string, "]");
  return debug_string;
}

void FlatSignature::PopulateNameToIndexMap() {
  all_with_names_ = true;
  for (int i = 0; i < specs_.size(); ++i) {
    if (specs_[i].name.has_value()) {
      name_to_index_[*specs_[i].name] = i;
    } else {
      all_with_names_ = false;
    }
  }
}

absl::Status FlatSignature::CheckIndexInRange(int index) const {
  if (index < 0 || index >= specs_.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The index ", index, "is out of range. It should be within [0, ",
        specs_.size() - 1, "]"));
  }
  return absl::OkStatus();
}

absl::Status FlatSignature::AddDefaultValue(absl::string_view name,
                                            Value default_value) {
  TF_ASSIGN_OR_RETURN(int index, GetIndexFromName(name));
  return AddDefaultValue(index, std::move(default_value));
}

absl::Status FlatSignature::AddDefaultValue(int index, Value default_value) {
  if (!default_value.HasValue()) {
    return absl::InvalidArgumentError(
        "The default value provided is not valid.");
  }
  TF_RETURN_IF_ERROR(CheckIndexInRange(index));
  if (specs_[index].default_value.HasValue()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The ArgSpec at ", index, " already has a default value."));
  }
  specs_[index].default_value = std::move(default_value);
  return absl::OkStatus();
}

absl::StatusOr<int> FlatSignature::GetIndexFromName(
    absl::string_view name) const {
  auto it = name_to_index_.find(name);
  if (it == name_to_index_.end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("The name \"", name, "\" is not found."));
  }
  // The index is guaranteed to be in range because the map is populated in the
  // constructor.
  return it->second;
}

// We pass `kwargs` by value to allow moving the values from it.
absl::Status FlatSignature::FillArgs(
    std::vector<Value>& args, absl::flat_hash_map<std::string, Value> kwargs,
    bool strict_match) const {
  if (args.size() > specs_.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("args.size(): ", args.size(),
                     " is larger than specs().size(): ", specs_.size()));
  }
  if (args.size() == specs_.size()) {
    if (kwargs.empty()) {
      return absl::OkStatus();
    }
    // Otherwise, there must be an error. Let the kwargs-related code handle it
    // and return a more specific error message.
  }
  if (args.empty() && !all_with_names_) {
    return absl::InvalidArgumentError(
        "Not all specs have names. Cannot flatten with only a map.");
  }

  int num_original_args = args.size();

  // The result will be a combination of the original `args` and `kwargs`. Here
  // we directly reuse the `args` vector to store the result to avoid
  // copying and will add elements to it.

  // Add the keyword arguments.
  args.resize(specs_.size());
  if (strict_match) {
    for (auto& [name, value] : kwargs) {
      TF_ASSIGN_OR_RETURN(int index, GetIndexFromName(name));
      if (args[index].HasValue()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Trying to set the value at name \"", name,
                         "\" with index ", index, " but it is already set."));
      }
      args[index] = std::move(value);
    }
    // Add the default values and check if all the values are set.
    for (int i = num_original_args; i < specs_.size(); ++i) {
      if (args[i].HasValue()) {
        continue;
      }
      if (!specs_[i].default_value.HasValue()) {
        return absl::InvalidArgumentError(
            absl::StrFormat("The value at index %d with name \"%s\" is not set "
                            "and does not have a default value.",
                            i, specs_[i].name.value_or("null")));
      }
      args[i] = specs_[i].default_value;
    }
  } else {
    for (int i = num_original_args; i < specs_.size(); ++i) {
      if (!specs_[i].name.has_value()) {
        return absl::InvalidArgumentError(
            absl::StrCat("The spec at index ", i, " does not have a name."));
      }
      const std::string& name = *specs_[i].name;
      auto it = kwargs.find(name);
      if (it != kwargs.end()) {
        args[i] = std::move(it->second);
      } else if (specs_[i].default_value.HasValue()) {
        args[i] = specs_[i].default_value;
      } else {
        return absl::InvalidArgumentError(
            absl::StrFormat("The value at index %d with name \"%s\" is not set "
                            "and does not have a default value.",
                            i, specs_[i].name.value_or("null")));
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<absl::flat_hash_map<std::string, Value>>
FlatSignature::UnflattenToMap(std::vector<Value> values) const {
  if (!all_with_names_) {
    return absl::FailedPreconditionError(
        "Some specs have no names. Cannot unflatten to a map.");
  }
  if (values.size() != specs_.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("The size of values ", values.size(),
                     " is not equal to the size of specs ", specs_.size()));
  }
  absl::flat_hash_map<std::string, Value> result;
  for (int i = 0; i < values.size(); ++i) {
    if (!specs_[i].name.has_value()) {
      // This should never happen because `all_with_names_` is true.
      return absl::InternalError(
          absl::StrCat("The spec at index ", i, " does not have a name."));
    }
    result[*specs_[i].name] = std::move(values[i]);
  }
  return result;
}

absl::StatusOr<std::vector<std::string>> FlatSignature::GetAllNames() const {
  if (!all_with_names_) {
    return absl::FailedPreconditionError(
        "Not all specs have names. Cannot get all names.");
  }
  std::vector<std::string> all_names;
  all_names.reserve(specs_.size());
  for (const auto& spec : specs_) {
    all_names.push_back(*spec.name);
  }
  return all_names;
}

}  // namespace neptune
