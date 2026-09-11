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

#include "runtime/core/value_utils.h"  // from @neptune

#include <utility>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "runtime/core/device_buffer.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune
#include "runtime/core/value.h"  // from @neptune
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla

namespace xla_serving {

absl::StatusOr<Value> Reshape(Value&& value,
                             const xla::ifrt::Shape& shape) {
  if (value.IsType<HostBuffer>()) {
    TF_ASSIGN_OR_RETURN(HostBuffer reshaped,
                        value.Get<HostBuffer>().Reshape(shape));
    return Value(std::move(reshaped));
  }
  if (value.IsType<DeviceBuffer>()) {
    TF_ASSIGN_OR_RETURN(DeviceBuffer reshaped,
                        std::move(value.Get<DeviceBuffer>()).Reshape(shape));
    return Value(std::move(reshaped));
  }
  return absl::InvalidArgumentError(
      "Value is neither a HostBuffer nor a DeviceBuffer.");
}

absl::StatusOr<Value> Bitcast(Value&& value, xla::ifrt::DType dtype,
                             const xla::ifrt::Shape& shape) {
  if (value.IsType<HostBuffer>()) {
    TF_ASSIGN_OR_RETURN(HostBuffer bitcast,
                        value.Get<HostBuffer>().Bitcast(dtype, shape));
    return Value(std::move(bitcast));
  }
  if (value.IsType<DeviceBuffer>()) {
    TF_ASSIGN_OR_RETURN(
        DeviceBuffer bitcast,
        std::move(value.Get<DeviceBuffer>()).Bitcast(dtype, shape));
    return Value(std::move(bitcast));
  }
  return absl::InvalidArgumentError(
      "Value is neither a HostBuffer nor a DeviceBuffer.");
}

}  // namespace xla_serving
