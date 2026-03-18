#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_GRID_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_GRID_H_

#include <ostream>
#include <string>

#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/strings/str_format.h"  // from @abseil-cpp

namespace jsv {

// Coordinates that identify a particular point in a 4-d grid (usually a TPU
// topology).
struct GridCoords {
  int dim[4];

  constexpr GridCoords(int d0, int d1, int d2, int d3) : dim{d0, d1, d2, d3} {}
  GridCoords() : GridCoords(0, 0, 0, 0) {}

  static GridCoords Zeroes() { return GridCoords(0, 0, 0, 0); }
  static GridCoords Ones() { return GridCoords(1, 1, 1, 1); }

  int operator[](int i) const {
    DCHECK_GE(i, 0);
    DCHECK_LT(i, 4);
    return dim[i];
  }

  int& operator[](int i) {
    DCHECK_GE(i, 0);
    DCHECK_LT(i, 4);
    return dim[i];
  }

  int Product() const { return dim[0] * dim[1] * dim[2] * dim[3]; }

  std::string ToString() const;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const GridCoords& value) {
    absl::Format(&sink, "%s", value.ToString());
  }

  friend bool operator==(const GridCoords& a, const GridCoords& b) {
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3];
  }

  friend std::ostream& operator<<(std::ostream& os, const GridCoords& c) {
    return os << c.ToString();
  }

  template <typename H>
  friend H AbslHashValue(H h, const GridCoords& c) {
    return H::combine(std::move(h), c[0], c[1], c[2], c[3]);
  }
};

}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_GRID_H_
