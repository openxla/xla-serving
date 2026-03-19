#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_TYPE_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_TYPE_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"  // from @abseil-cpp
#include "runtime/core/value.h"  // from @neptune

namespace neptune::type {

/* A structured-data (a.k.a. "tree") system (to e.g. represent function
signatures).

We use `Tree` to store structured data. There are only three cases of `Tree`
which are special and treated as "structural":

- `Tuple`: A list of `Tree`s.
- `Dict`: A dictionary (i.e. map) from string to `Tree`.
- `None`: A special Tree that represents "nothing". (It's isomorphic to an empty
`Tuple` or an empty `Dict`, but we define it explicitly to align with Python.)

All other cases are treated as "leaf" (note that `None` is not treated as a
leaf).
*/

// We reuse the type-erased `Value` as `Tree`, for convenience. We could have
// defined a new type for `Tree` as:
//
//   struct Tree {
//     std::variant<std::vector<Tree>, std::map<std::string, Tree>, ...> tree;
//   }
//
// TODO(wangpeng): Consider defining a new type for `Tree`.
using Tree = Value;

using Tuple = std::vector<Tree>;

// We choose `absl::btree_map` because it's ordered, which is convenient for us
// because we often need to iterate over the dict in key order (e.g. for
// flattening and unflattening). We could have chosen an unordered map and
// manually sort by key when needed, but that would be more toil.
using Dict = absl::btree_map<std::string, Tree>;

struct None {};

// The difference between `Value` and `Type` is that `Value`'s
// leaves should be things like `HostBuffer` or `tensorflow::Tensor` while
// `Type`'s leaves should be things like `ArgSpec`.
//
// `Type` is isomorphic to `orbax_model_type::Type`.
using Type = Tree;

// A list of (key, value) pairs for `Tree`.
using StringTypePairs = std::vector<std::pair<std::string, Type>>;

}  // namespace neptune::type

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_TYPE_H_
