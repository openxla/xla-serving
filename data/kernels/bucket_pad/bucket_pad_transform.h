// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef XLA_SERVING_DATA_KERNELS_BUCKET_PAD_BUCKET_PAD_TRANSFORM_H_
#define XLA_SERVING_DATA_KERNELS_BUCKET_PAD_BUCKET_PAD_TRANSFORM_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/kernels/bucket_pad/bucket_pad_transform.pb.h"  // from @xla_serving
#include "data/processor/cc/api/transform_context.h"  // from @xla_serving
#include "data/processor/cc/api/type_info.h"  // from @xla_serving
#include "data/processor/cc/core/transform.h"  // from @xla_serving
#include "data/processor/cc/utils/default_value.pb.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving

namespace xla_serving::data::composite::transforms {

// A transform that dynamically pads a tensor along a specified axis to the
// smallest pre-determined bucket dimension D_fixed >= C from target_dimensions.
class BucketPadTransform : public TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<BucketPadTransform>> Create(
      std::vector<int64_t> target_dimensions, int32_t axis = -1,
      utils::DefaultValue pad_value = {});

  static absl::StatusOr<std::unique_ptr<BucketPadTransform>> CreateFromProto(
      const BucketPadArgs& args);

  BucketPadTransform(std::vector<int64_t> target_dimensions, int32_t axis,
                     utils::DefaultValue pad_value);

  absl::Status Execute(api::ExecuteContext& context) override;

  absl::Status PropagateSignatureImpl(
      api::ShapeInferenceContext& context) const override;

  absl::Span<const api::TypeInfo> InputTypes() const override {
    return input_types_;
  }

  absl::Span<const api::TypeInfo> OutputTypes() const override {
    return output_types_;
  }

  absl::StatusOr<::xla_serving::data::composite::TransformConfig> ToProto()
      const override;

  absl::StatusOr<xla_serving::HostBuffer> operator()(
      const xla_serving::HostBuffer& input) const;

  std::string DebugName() const override;

  absl::Span<const int64_t> target_dimensions() const {
    return target_dimensions_;
  }
  int32_t axis() const { return axis_; }
  const utils::DefaultValue& pad_value() const { return pad_value_; }

 private:
  std::vector<int64_t> target_dimensions_;
  int32_t axis_;
  utils::DefaultValue pad_value_;
  std::vector<api::TypeInfo> input_types_;
  std::vector<api::TypeInfo> output_types_;
};

}  // namespace xla_serving::data::composite::transforms

#endif  // XLA_SERVING_DATA_KERNELS_BUCKET_PAD_BUCKET_PAD_TRANSFORM_H_
