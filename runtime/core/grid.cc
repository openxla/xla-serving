#include "third_party/neptune/runtime/core/grid.h"

#include <string>

#include "third_party/absl/strings/str_cat.h"

namespace jsv {

std::string GridCoords::ToString() const {
  return absl::StrCat("[", dim[0], ",", dim[1], ",", dim[2], ",", dim[3], "]");
}

}  // namespace jsv
