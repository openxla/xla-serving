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

#ifndef XLA_SERVING_DATA_PROCESSOR_PYTHON_EXAMPLES_INT_RESOURCE_H_
#define XLA_SERVING_DATA_PROCESSOR_PYTHON_EXAMPLES_INT_RESOURCE_H_

#include "absl/status/status.h"  // from @abseil-cpp
#include "data/resource_manager/cc/resource.h"  // from @xla_serving

namespace xla_serving::data {
namespace composite {

class IntResource : public Resource {
 public:
  using Resource::Resource;
  absl::Status LoadImpl() override;
  int value() const { return value_; }

 private:
  int value_ = 0;
};

class IntContentResource : public Resource {
 public:
  using Resource::Resource;
  absl::Status LoadImpl() override;
  int value() const { return value_; }

 private:
  int value_ = 0;
};

}  // namespace composite
}  // namespace xla_serving::data

#endif  // XLA_SERVING_DATA_PROCESSOR_PYTHON_EXAMPLES_INT_RESOURCE_H_
