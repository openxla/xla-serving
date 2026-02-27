#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_FLAT_SIGNATURE_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_FLAT_SIGNATURE_H_

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/neptune/runtime/core/stablehlo_utils.h"
#include "third_party/neptune/runtime/core/value.h"
#include "third_party/py/orbax/experimental/model/core/protos/manifest.proto.h"

namespace jsv {

// An enhanced version of `ArgSpec` that contains more information about the
// argument.
struct MetaArgSpec {
 public:
  std::string DebugString() const;

  // The argument specification.
  ArgSpec spec;
  // The argument name. Kwarg-only arguments must have names. Some positional
  // arguments have names. For example, In TensorFlow, we save the names for
  // even the positional arguments because they are needed to call the function
  // with a map.
  std::optional<std::string> name;
  // We want a default value so we can fill in the missing arguments when
  // calling the function with a map.
  // Use `Value::HasValue()` to check if the default value is set.
  Value default_value;

  // Check if two `MetaArgSpec` are equal except the default value. This
  // function checks the existence of the default value, but not the value
  // itself.
  bool EqualExceptDefaultValue(const MetaArgSpec& rhs) const;
};

// A flat function signature that contains a list of argument specifications.
// The argument specifications can be positional or by name. This can be seen as
// a plugin to allow calling a function with the Python function call syntax:
// for the same argument, you may pass it as positional argument, or use
// keyword. The key of this class is to allow converting args-kwargs to a list
// of values (for input), and convert the list of values back to a map.
class FlatSignature {
 public:
  explicit FlatSignature(std::vector<MetaArgSpec> specs)
      : specs_(std::move(specs)) {
    PopulateNameToIndexMap();
  }

  // Returns the argument specifications.
  const std::vector<MetaArgSpec>& specs() const { return specs_; }

  std::string DebugString() const;

  // Adds a default value to the argument specification with the given name.
  // Returns an error if the name is not found or there is already a default
  // value.
  absl::Status AddDefaultValue(absl::string_view name, Value default_value);

  // Same as above, but with the index of the argument specification.
  absl::Status AddDefaultValue(int index, Value default_value);

  // Fills in the keyword arguments to possibly extend the vector of `args`. The
  // original elements in `args` will be put at the beginning of the list. The
  // kwargs will fill in the missing arguments based on their names and the
  // corresponding names in `specs_`.
  absl::Status FillArgs(std::vector<Value>& args,
                        absl::flat_hash_map<std::string, Value> kwargs) const;

  // Unflattens a list of values to a map. The input `Value`s will be moved to
  // the output.
  absl::StatusOr<absl::flat_hash_map<std::string, Value>> UnflattenToMap(
      std::vector<Value> values) const;

  // Returns true if all the argument specifications have names.
  bool IsAllSpecWithName() const { return all_with_names_; }

  // Returns all the names of the argument specifications. If
  // `IsAllSpecWithName()` is false, returns an error.
  absl::StatusOr<std::vector<std::string>> GetAllNames() const;

 private:
  // Populates the `name_to_index_` map.
  void PopulateNameToIndexMap();

  // Checks if the index is in range.
  absl::Status CheckIndexInRange(int index) const;

  // Returns the index of the argument specification with the given name.
  // Returns an error if the name is not found or the index is out of range.
  absl::StatusOr<int> GetIndexFromName(absl::string_view name) const;

  std::vector<MetaArgSpec> specs_;

  // This helps to call the function with a map.
  absl::flat_hash_map<std::string, int> name_to_index_;

  // This is true if all the argument specifications have names.
  bool all_with_names_ = false;
};

// A flat function signature that contains both input and output signatures.
struct FunctionFlatSignature {
  FlatSignature input_signature;
  FlatSignature output_signature;
};

}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_FLAT_SIGNATURE_H_
