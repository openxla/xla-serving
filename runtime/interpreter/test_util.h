#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_TEST_UTIL_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_TEST_UTIL_H_

#include "testing/base/public/gmock.h"

namespace jsv::test_util {

// Checks that the flat signature matches the expected meta arg specs in the
// same order. `meta_arg_specs` is a vector of `MetaArgSpec`.
MATCHER_P(MetaArgSpecsAre, meta_arg_specs,
          "matches the expected meta arg specs in the same order.") {
  if (arg.specs().size() != meta_arg_specs.size()) {
    return false;
  }
  for (int i = 0; i < arg.specs().size(); ++i) {
    if (!arg.specs()[i].EqualExceptDefaultValue(meta_arg_specs[i])) {
      return false;
    }
  }
  return true;
}

}  // namespace jsv::test_util

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_TEST_UTIL_H_
