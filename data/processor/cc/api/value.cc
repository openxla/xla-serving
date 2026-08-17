// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
