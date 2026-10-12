/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Copyright 2026 Google LLC. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "data/kernels/math/reduction_ops_common.h"  // from @xla_serving
#include "data/processor/cc/api/registry.h"  // from @xla_serving

namespace xla_serving::data::kernel {

ND_REGISTER_TRANSFORM(ReduceMeanTransform, "ReduceMeanTransform")
    .SetDeserializer(&DeserializeReductionTransform<ReduceMeanTransform>);

}  // namespace xla_serving::data::kernel
