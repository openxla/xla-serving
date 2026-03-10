#include "runtime/core/grid.h"  // from @neptune

#include <string>

#include "absl/strings/str_cat.h"  // from @com_google_absl

namespace jsv {

std::string GridCoords::ToString() const {
  return absl::StrCat("[", dim[0], ",", dim[1], ",", dim[2], ",", dim[3], "]");
}

}  // namespace jsv
