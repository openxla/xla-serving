#include "data/processor/python/examples/int_resource.h"  // from @neptune

#include <fstream>
#include <sstream>
#include <string>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/strings/numbers.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "data/resource_manager/cc/resource.h"  // from @neptune

namespace neptune::data {
namespace composite {

absl::Status IntResource::LoadImpl() {
  if (assets().empty() || assets()[0].path.empty()) {
    return absl::InvalidArgumentError(
        "IntResource requires an asset with a non-empty path.");
  }
  const std::string& path = assets()[0].path;

  std::ifstream file(path);
  if (!file) {
    return absl::InvalidArgumentError(
        absl::StrCat("No asset found at specified path: ", path));
  }

  std::ostringstream ss;
  ss << file.rdbuf();
  std::string content = ss.str();

  if (!absl::SimpleAtoi(content, &value_)) {
    return absl::InvalidArgumentError(
        "Could not parse asset content as integer.");
  }
  return absl::OkStatus();
}

ND_REGISTER_RESOURCE(IntResource, "IntResource")

absl::Status IntContentResource::LoadImpl() {
  if (assets().empty() || assets()[0].content.empty()) {
    return absl::InvalidArgumentError(
        "IntContentResource requires an asset with a non-empty content.");
  }
  const std::string& content = assets()[0].content;

  if (!absl::SimpleAtoi(content, &value_)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Could not parse asset content as integer: ", content));
  }
  return absl::OkStatus();
}

ND_REGISTER_RESOURCE(IntContentResource, "IntContentResource")

}  // namespace composite
}  // namespace neptune::data
