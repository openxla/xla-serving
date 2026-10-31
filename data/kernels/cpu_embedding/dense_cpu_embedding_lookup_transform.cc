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

#include "data/kernels/cpu_embedding/dense_cpu_embedding_lookup_transform.h"  // from @xla_serving

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/cpu_embedding/cc/cpu_embedding_engine.h"  // from @xla_serving
#include "data/kernels/cpu_embedding/cpu_embedding_resource.h"  // from @xla_serving
#include "data/kernels/cpu_embedding/dense_cpu_embedding_lookup.pb.h"  // from @xla_serving
#include "data/kernels/cpu_embedding/safe_int_utils.h"  // from @xla_serving
#include "data/processor/cc/api/registry.h"  // from @xla_serving
#include "data/processor/cc/api/shape.h"  // from @xla_serving
#include "data/processor/cc/api/transform_context.h"  // from @xla_serving
#include "data/processor/cc/api/type_info.h"  // from @xla_serving
#include "data/processor/cc/core/transform.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "data/resource_manager/cc/resource.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla

namespace xla_serving::data::composite {
namespace {

struct PreparedLookup {
  std::vector<int64_t> active_ids;
  std::vector<int32_t> row_pointers;
};

}  // namespace

DenseCpuEmbeddingLookupTransform::DenseCpuEmbeddingLookupTransform(
    std::shared_ptr<CpuEmbeddingEngineResource> resource,
    std::vector<std::string> feature_names, int64_t pad_value,
    std::vector<std::string> output_feature_names)
    : resource_(std::move(resource)),
      feature_names_(std::move(feature_names)),
      pad_value_(pad_value),
      output_feature_names_(std::move(output_feature_names)) {
  for (size_t i = 0; i < feature_names_.size(); ++i) {
    input_types_.push_back(api::CreateTypeInfo<xla_serving::HostBuffer>());
    output_types_.push_back(api::CreateTypeInfo<xla_serving::HostBuffer>());
  }
  debug_string_ = absl::StrCat(
      "DenseCpuEmbeddingLookupTransform(features=[",
      absl::StrJoin(feature_names_, ", "), "], pad_value=", pad_value_,
      ", resource=", (resource_ ? absl::StrCat(resource_->id()) : "null"), ")");
}

absl::StatusOr<std::unique_ptr<DenseCpuEmbeddingLookupTransform>>
DenseCpuEmbeddingLookupTransform::Create(
    std::shared_ptr<CpuEmbeddingEngineResource> resource,
    std::vector<std::string> feature_names, int64_t pad_value,
    std::vector<std::string> output_feature_names) {
  if (resource == nullptr) {
    return absl::InvalidArgumentError(
        "DenseCpuEmbeddingLookupTransform requires a non-null resource.");
  }
  if (feature_names.empty()) {
    return absl::InvalidArgumentError(
        "DenseCpuEmbeddingLookupTransform requires at least one feature name.");
  }
  for (const auto& name : feature_names) {
    if (name.empty()) {
      return absl::InvalidArgumentError(
          "DenseCpuEmbeddingLookupTransform requires non-empty feature names.");
    }
  }
  return absl::WrapUnique(
      new DenseCpuEmbeddingLookupTransform(std::move(resource),
                                           std::move(feature_names), pad_value,
                                           std::move(output_feature_names)));
}

absl::StatusOr<std::unique_ptr<DenseCpuEmbeddingLookupTransform>>
DenseCpuEmbeddingLookupTransform::CreateForFeature(
    std::shared_ptr<CpuEmbeddingEngineResource> resource,
    std::string feature_name, int64_t pad_value) {
  return Create(std::move(resource),
                std::vector<std::string>{std::move(feature_name)}, pad_value,
                {});
}

std::vector<std::shared_ptr<xla_serving::data::Resource>>
DenseCpuEmbeddingLookupTransform::GetAllResources() const {
  return {resource_};
}

absl::Span<const api::TypeInfo> DenseCpuEmbeddingLookupTransform::InputTypes()
    const {
  return absl::MakeConstSpan(input_types_);
}

absl::Span<const api::TypeInfo> DenseCpuEmbeddingLookupTransform::OutputTypes()
    const {
  return absl::MakeConstSpan(output_types_);
}

absl::Status DenseCpuEmbeddingLookupTransform::Execute(
    api::ExecuteContext& context) {
  if (context.inputs.size() != feature_names_.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "DenseCpuEmbeddingLookupTransform expected ", feature_names_.size(),
        " inputs, got ", context.inputs.size()));
  }

  std::vector<xla_serving::HostBuffer> host_buffers;
  host_buffers.reserve(context.inputs.size());
  for (size_t i = 0; i < context.inputs.size(); ++i) {
    host_buffers.push_back(context.inputs[i].Get<xla_serving::HostBuffer>());
  }

  ASSIGN_OR_RETURN(std::vector<xla_serving::HostBuffer> outputs,
                   (*this)(std::move(host_buffers)));

  for (auto& out_hb : outputs) {
    context.outputs.push_back(api::Value(std::move(out_hb)));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<xla_serving::HostBuffer>>
DenseCpuEmbeddingLookupTransform::operator()(
    std::vector<xla_serving::HostBuffer> dense_ids_list) {
  if (dense_ids_list.size() != feature_names_.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "DenseCpuEmbeddingLookupTransform expected ", feature_names_.size(),
        " inputs, got ", dense_ids_list.size()));
  }

  std::vector<PreparedLookup> prepared(feature_names_.size());

  for (size_t f = 0; f < feature_names_.size(); ++f) {
    const auto& dense_ids = dense_ids_list[f];
    if (dense_ids.shape().dims().empty()) {
      return absl::InvalidArgumentError(absl::StrCat("dense_ids for feature ",
                                                     feature_names_[f],
                                                     " must have rank >= 1."));
    }
    int64_t num_elements = dense_ids.shape().num_elements();
    if (num_elements > 0 && dense_ids.base() == nullptr) {
      return absl::InvalidArgumentError(absl::StrCat(
          "dense_ids base pointer for feature ", feature_names_[f],
          " is null with ", num_elements, " elements."));
    }
    if (num_elements > std::numeric_limits<int32_t>::max()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "dense_ids for feature ", feature_names_[f],
          " exceeds INT32_MAX elements: ", num_elements));
    }
    int64_t batch_size = dense_ids.shape().dims()[0];
    if (batch_size > std::numeric_limits<int32_t>::max()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "batch_size for feature ", feature_names_[f],
          " exceeds INT32_MAX: ", batch_size));
    }
    int64_t sequence_length =
        batch_size > 0 ? num_elements / batch_size : 0;
    if (sequence_length > std::numeric_limits<int32_t>::max()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "sequence_length for feature ", feature_names_[f],
          " exceeds INT32_MAX: ", sequence_length));
    }

    std::vector<int64_t>& active_ids = prepared[f].active_ids;
    active_ids.reserve(num_elements);
    std::vector<int32_t>& row_pointers = prepared[f].row_pointers;
    row_pointers.reserve(batch_size + 1);
    row_pointers.push_back(0);

    auto process_ids = [&](const auto* id_ptr) -> absl::Status {
      for (int64_t i = 0; i < batch_size; ++i) {
        int32_t active_count = 0;
        for (int64_t j = 0; j < sequence_length; ++j) {
          int64_t id = id_ptr[i * sequence_length + j];
          if (id != pad_value_) {
            active_ids.push_back(id);
            active_count++;
          }
        }
        if (active_count == 0) {
          active_ids.push_back(pad_value_);
          active_count = 1;
        }
        int32_t next_offset = 0;
        if (!xla_serving::data::utils::SafeAdd(row_pointers.back(), active_count,
                                           &next_offset)) {
          return absl::InvalidArgumentError(
              "Row pointer offset overflowed int32_t.");
        }
        row_pointers.push_back(next_offset);
      }
      return absl::OkStatus();
    };

    if (dense_ids.dtype().kind() == xla::ifrt::DType::Kind::kS64) {
      RETURN_IF_ERROR(
          process_ids(reinterpret_cast<const int64_t*>(dense_ids.base())));
    } else if (dense_ids.dtype().kind() == xla::ifrt::DType::Kind::kS32) {
      RETURN_IF_ERROR(
          process_ids(reinterpret_cast<const int32_t*>(dense_ids.base())));
    } else {
      return absl::InvalidArgumentError(absl::StrCat(
          "dense_ids for feature ", feature_names_[f],
          " must be kS64 or kS32, got kind: ",
          static_cast<int>(dense_ids.dtype().kind())));
    }
  }

  // Borrow the CSR built above and have the engine write each feature's
  // [batch_size, emb_dim] result directly into its output HostBuffer.
  using FeatureBatchInput =
      cpu_embedding::CpuEmbeddingEngine::FeatureBatchInput<float>;
  std::vector<xla_serving::HostBuffer> output_buffers(feature_names_.size());
  std::vector<FeatureBatchInput> batch_inputs;
  batch_inputs.reserve(feature_names_.size());
  for (size_t f = 0; f < feature_names_.size(); ++f) {
    const int64_t batch_size = dense_ids_list[f].shape().dims()[0];
    ASSIGN_OR_RETURN(int32_t emb_dim,
                     resource_->GetEmbeddingDim(feature_names_[f]));
    batch_inputs.push_back({
        .feature_name = feature_names_[f],
        .embedding_ids = prepared[f].active_ids,
        .row_pointers = prepared[f].row_pointers,
        .weights = {},
        .output_allocator =
            [&output_buffers, f, batch_size,
             emb_dim](size_t /*requested_elements*/) {
              // The engine rejects the span if its size differs from the
              // requested size.
              output_buffers[f] = xla_serving::HostBuffer(
                  xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
                  xla::ifrt::Shape({batch_size, emb_dim}));
              auto output = output_buffers[f].flat<float>();
              return absl::MakeSpan(output.data(), output.size());
            },
    });
  }

  RETURN_IF_ERROR(resource_->Get()->ExecuteBatch(absl::MakeSpan(batch_inputs)));
  return output_buffers;
}

absl::StatusOr<TransformConfig> DenseCpuEmbeddingLookupTransform::ToProto()
    const {
  TransformConfig config;
  config.set_op_name("DenseCpuEmbeddingLookupTransform");
  if (!feature_names_.empty()) {
    config.set_name(feature_names_[0]);
  }
  if (resource_ != nullptr) {
    config.add_resource_ids(resource_->id());
  }

  transforms::cpu_embedding::DenseCpuEmbeddingLookupArgs args;
  args.mutable_feature_names()->Assign(feature_names_.begin(),
                                       feature_names_.end());
  args.set_pad_value(pad_value_);
  args.mutable_output_feature_names()->Assign(output_feature_names_.begin(),
                                              output_feature_names_.end());
  if (!feature_names_.empty()) {
    args.set_feature_name(feature_names_[0]);
  }
  config.mutable_detail()->PackFrom(args);

  return config;
}

ND_REGISTER_TRANSFORM(DenseCpuEmbeddingLookupTransform,
                      "DenseCpuEmbeddingLookupTransform")
    .SetDeserializerWithDetail<
        transforms::cpu_embedding::DenseCpuEmbeddingLookupArgs>(
        [](const transforms::cpu_embedding::DenseCpuEmbeddingLookupArgs& args,
           absl::Span<const std::shared_ptr<xla_serving::data::Resource>> resources)
            -> absl::StatusOr<std::unique_ptr<TransformBase>> {
          if (resources.empty()) {
            return absl::InvalidArgumentError(
                "DenseCpuEmbeddingLookupTransform requires a resource.");
          }
          auto resource = std::dynamic_pointer_cast<CpuEmbeddingEngineResource>(
              resources[0]);
          if (resource == nullptr) {
            return absl::InvalidArgumentError(
                "Failed to cast resource to CpuEmbeddingEngineResource.");
          }

          std::vector<std::string> feature_names(args.feature_names().begin(),
                                                 args.feature_names().end());
          if (feature_names.empty()) {
            if (args.has_feature_name() && !args.feature_name().empty()) {
              feature_names.push_back(args.feature_name());
            } else {
              return absl::InvalidArgumentError(
                  "DenseCpuEmbeddingLookupTransform requires at least one "
                  "feature name.");
            }
          }
          std::vector<std::string> output_feature_names(
              args.output_feature_names().begin(),
              args.output_feature_names().end());
          return DenseCpuEmbeddingLookupTransform::Create(
              std::move(resource), std::move(feature_names), args.pad_value(),
              std::move(output_feature_names));
        });

absl::Status DenseCpuEmbeddingLookupTransform::PropagateSignatureImpl(
    api::ShapeInferenceContext& context) const {
  if (context.inputs.size() != feature_names_.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "DenseCpuEmbeddingLookupTransform expects ", feature_names_.size(),
        " inputs, got ", context.inputs.size()));
  }

  for (size_t f = 0; f < feature_names_.size(); ++f) {
    if (!context.inputs[f].IsSpec()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Input ", f, " is not a HostBufferSpec."));
    }
    const auto& ids_spec = context.inputs[f].spec();
    ASSIGN_OR_RETURN(int32_t embedding_dim,
                     resource_->GetEmbeddingDim(feature_names_[f]));
    const api::Dim batch_size = ids_spec.shape.dims().empty()
                                    ? api::Dim::Unknown()
                                    : ids_spec.shape.dims()[0];
    ASSIGN_OR_RETURN(const api::Dim embedding_size,
                     api::Dim::FromSize(embedding_dim));
    ASSIGN_OR_RETURN(api::PartialShape out_shape,
                     api::PartialShape::Create({batch_size, embedding_size}));

    context.outputs.push_back(api::ShapeInferenceValue(api::HostBufferSpec{
        xla::ifrt::DType(xla::ifrt::DType::Kind::kF32), std::move(out_shape)}));
  }
  return absl::OkStatus();
}

}  // namespace xla_serving::data::composite
