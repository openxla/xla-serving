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
