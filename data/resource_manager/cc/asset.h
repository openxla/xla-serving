#ifndef THIRD_PARTY_NEPTUNE_DATA_RESOURCE_MANAGER_CC_ASSET_H_
#define THIRD_PARTY_NEPTUNE_DATA_RESOURCE_MANAGER_CC_ASSET_H_

#include <string>

#include "absl/status/status.h"  // from @abseil-cpp

namespace neptune::data {

struct Asset {
  const std::string name;
  const std::string path;

  // Primarily for pathways support when assets are not available on each
  // worker. We can remove if this is deemed unnecessary.
  std::string content = "";
  absl::Status LoadContent();

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Asset& asset) {
    absl::Format(&sink, "Asset(name=%s, path=%s, content_size=%zu)", asset.name,
                 asset.path, asset.content.size());
  }
};

}  // namespace neptune::data

#endif  // THIRD_PARTY_NEPTUNE_DATA_RESOURCE_MANAGER_CC_ASSET_H_
