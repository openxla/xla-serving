#include "runtime/data_loader/ocp_loader/metadata_util.h"  // from @neptune

#include <string>
#include <vector>

#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "third_party/json/include/nlohmann/json.hpp"
#include "third_party/json/include/nlohmann/json_fwd.hpp"
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "tsl/platform/path.h"  // from @tsl
#include "util/task/status_macros.h"

namespace neptune {

constexpr const char* METADATA_FILE_NAME = "_METADATA";
constexpr const char* TREE_METADATA = "tree_metadata";
constexpr const char* KEY_METADATA = "key_metadata";
constexpr const char* VALUE_METADATA = "value_metadata";
constexpr const char* SKIP_DESERIALIZE = "skip_deserialize";
constexpr const char* KEY = "key";

absl::StatusOr<std::vector<std::vector<std::string>>> GetParamNameLists(
    const nlohmann::ordered_json& metadata_json) {
  if (!metadata_json.contains(TREE_METADATA)) {
    return absl::InvalidArgumentError(
        "Input metadata json does not contain tree_metadata");
  }
  nlohmann::ordered_json tree_metadata = metadata_json[TREE_METADATA];
  std::vector<std::vector<std::string>> param_name_lists;
  for (nlohmann::ordered_json::iterator tree_metadata_iter =
           tree_metadata.begin();
       tree_metadata_iter != tree_metadata.end(); ++tree_metadata_iter) {
    nlohmann::ordered_json tree_metadata_kv = tree_metadata_iter.value();
    if (!tree_metadata_kv.contains(KEY_METADATA)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Empty key_metadata: ", tree_metadata_kv.dump()));
    }
    if (!tree_metadata_kv.contains(VALUE_METADATA) ||
        !tree_metadata_kv[VALUE_METADATA].contains(SKIP_DESERIALIZE)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid value_metadata: ", tree_metadata_kv.dump()));
    }
    bool skip_deserialize =
        tree_metadata_kv[VALUE_METADATA][SKIP_DESERIALIZE].get<bool>();
    if (skip_deserialize) {
      continue;
    }
    nlohmann::ordered_json key_metadata =
            tree_metadata_kv[KEY_METADATA];
    std::vector<std::string> paths;
    for (nlohmann::ordered_json::iterator key_metadata_iter =
             key_metadata.begin();
         key_metadata_iter != key_metadata.end(); ++key_metadata_iter) {
      nlohmann::ordered_json key_meta = key_metadata_iter.value();
      if (key_meta.contains(KEY)) {
        paths.push_back(key_meta[KEY]);
      }
    }
    if (paths.empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat("key_metadata contains no keys: ", key_metadata.dump()));
    }
    param_name_lists.push_back(paths);
  }
  return param_name_lists;
}

std::vector<std::string> GetParamNamesFromLists(
    const std::vector<std::vector<std::string>>& param_name_lists) {
  std::vector<std::string> param_names;
  param_names.reserve(param_name_lists.size());
  for (const auto& param_name_list : param_name_lists) {
    param_names.push_back(absl::StrJoin(param_name_list, "."));
  }
  return param_names;
}

absl::StatusOr<std::vector<std::string>> GetParamNames(
    const nlohmann::ordered_json& metadata_json) {
  ASSIGN_OR_RETURN(std::vector<std::vector<std::string>> param_name_lists,
                   GetParamNameLists(metadata_json));
  std::vector<std::string> param_names =
      GetParamNamesFromLists(param_name_lists);
  return param_names;
}

bool is_prefix_path(const std::vector<std::string>& sub_tree_path,
                    const std::vector<std::string>& full_path) {
  if (sub_tree_path.size() > full_path.size()) {
    return false;
  }
  for (int i = 0; i < sub_tree_path.size(); ++i) {
    if (sub_tree_path[i] != full_path[i]) {
      return false;
    }
  }
  return true;
}

absl::StatusOr<std::vector<std::string>> GetParamNamesForSubTree(
    const nlohmann::ordered_json& metadata_json,
    std::vector<std::string>& sub_tree) {
  ASSIGN_OR_RETURN(std::vector<std::vector<std::string>> param_name_lists,
                   GetParamNameLists(metadata_json));
  if (sub_tree.empty()) {  // Fall back to get all param names
    return GetParamNamesFromLists(param_name_lists);
  }
  if (param_name_lists.empty()) {
    return std::vector<std::string>();
  }

  std::vector<std::string> filtered_param_names;
  for (const auto& param_name_list : param_name_lists) {
    if (is_prefix_path(sub_tree, param_name_list)) {
      filtered_param_names.push_back(absl::StrJoin(param_name_list, "."));
    }
  }
  if (filtered_param_names.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("No valid param names found for sub_tree path: ",
                     absl::StrJoin(sub_tree, "."),
                     " (Did you misspell something in sub_tree?)"));
  }
  return filtered_param_names;
}

absl::StatusOr<std::vector<std::string>> GetAllWeightNamesFromCheckpoint(
    absl::string_view checkpoint_path) {
  // Parse _METADATA to get the path to each weight.
  std::string metadata;
  TF_RETURN_IF_ERROR(tsl::ReadFileToString(
      tsl::Env::Default(), tsl::io::JoinPath(checkpoint_path, METADATA_FILE_NAME),
      &metadata));
  VLOG(1) << "metadata_json: " << metadata;
  nlohmann::ordered_json metadata_json =
      nlohmann::ordered_json::parse(metadata);
  // TODO: For now this assumes _METADATA file contains only
  // weights needed for inference. In reality, the OCP _METADATA file may
  // contain other unrelated keys (in terms of inference) such as optimizer
  // state. We will need OCP view proto to filter out non-weights keys.
  TF_ASSIGN_OR_RETURN(auto all_weight_names, GetParamNames(metadata_json));
  LOG(INFO) << "Number of weights: " << all_weight_names.size();
  return all_weight_names;
}

}  // namespace neptune
