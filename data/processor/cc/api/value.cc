#include "data/processor/cc/api/value.h"  // from @neptune

#include <string>
#include <utility>

#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "data/processor/cc/api/type_info.h"  // from @neptune

namespace neptune::data {
namespace composite {
namespace api {

TypeInfo Value::GetTypeInfo() const {
  return handler_(value_internal::Action::kTypeInfo, const_cast<Value*>(this),
                  nullptr);
}

std::string Value::DebugString() const {
  if (!HasValue()) return "(empty Value)";
  Value result;
  handler_(value_internal::Action::kDebugString, const_cast<Value*>(this),
           &result);
  return absl::StrCat("Value(", std::move(result.Get<std::string>()), ")");
}

}  // namespace api
}  // namespace composite
}  // namespace neptune::data
