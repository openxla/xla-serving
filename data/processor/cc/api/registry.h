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

#ifndef THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_CC_API_REGISTRY_H_
#define THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_CC_API_REGISTRY_H_

#include "data/processor/cc/core/transform_registry.h"  // from @neptune  // IWYU pragma: keep

#define ND_REGISTER_TRANSFORM_INTERNAL_IMPL(Type, Name, Id)         \
  static ::neptune::data::composite::core::TransformRegistrar<Type> \
      Transform_Registered_##Id =                                   \
          ::neptune::data::composite::core::TransformRegistrar<Type>(Name)

#define ND_REGISTER_TRANSFORM_INTERNAL(Type, Name, Id) \
  ND_REGISTER_TRANSFORM_INTERNAL_IMPL(Type, Name, Id)

#define ND_REGISTER_TRANSFORM(Type, Name) \
  ND_REGISTER_TRANSFORM_INTERNAL(Type, Name, __COUNTER__)

#endif  // THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_CC_API_REGISTRY_H_
