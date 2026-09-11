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

#include "data/kernels/cpu_embedding/cpu_embedding_lookup_transform.h"  // from @neptune

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "xla/tsl/platform/status_macros.h"
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_format.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/kernels/cpu_embedding/cpu_embedding.pb.h"  // from @neptune
#include "data/kernels/cpu_embedding/cpu_embedding_resource.h"  // from @neptune
#include "data/processor/cc/api/registry.h"  // from @neptune
#include "data/processor/cc/api/transform_context.h"  // from @neptune
#include "data/processor/cc/core/transform.h"  // from @neptune
#include "data/processor/proto/transform_config.pb.h"  // from @neptune
#include "data/resource_manager/cc/resource.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune
#include "xla/python/ifrt/dtype.h"  // from @xla

namespace neptune::data {
namespace composite {

namespace {}  // namespace

CpuEmbeddingLookupTransform::CpuEmbeddingLookupTransform(
    std::shared_ptr<CpuEmbeddingEngineResource> resource,
    std::vector<std::string> feature_names,
    std::vector<std::string> output_feature_names, bool dense_lookup,
    int64_t embedding_dim, std::vector<std::vector<int64_t>> output_shapes)
    : resource_(std::move(resource)),
      feature_names_(std::move(feature_names)),
      output_feature_names_(std::move(output_feature_names)),
      dense_lookup_(dense_lookup),
      embedding_dim_(embedding_dim),
      output_shapes_(std::move(output_shapes)) {}

absl::StatusOr<std::unique_ptr<CpuEmbeddingLookupTransform>>
CpuEmbeddingLookupTransform::Create(
    std::shared_ptr<CpuEmbeddingEngineResource> resource,
    std::vector<std::string> feature_names, bool dense_lookup,
    std::vector<std::string> output_feature_names, int64_t embedding_dim,
    std::vector<std::vector<int64_t>> output_shapes) {
  if (resource == nullptr) {
    return absl::InvalidArgumentError(
        "CpuEmbeddingLookupTransform requires a non-null resource.");
  }
  if (feature_names.empty()) {
    return absl::InvalidArgumentError(
        "CpuEmbeddingLookupTransform requires at least one feature name.");
  }
  if (!output_feature_names.empty() &&
      output_feature_names.size() != feature_names.size()) {
    return absl::InvalidArgumentError(
        "CpuEmbeddingLookupTransform output_feature_names size must match "
        "feature_names size if provided.");
  }
  absl::flat_hash_set<absl::string_view> unique_features;
  unique_features.reserve(feature_names.size());
  for (const auto& name : feature_names) {
    if (name.empty()) {
      return absl::InvalidArgumentError(
          "CpuEmbeddingLookupTransform requires non-empty feature names.");
    }
    if (!unique_features.insert(name).second) {
      return absl::InvalidArgumentError(
          absl::StrCat("Duplicate feature name found: ", name));
    }
  }

  if (!output_feature_names.empty()) {
    if (output_feature_names.size() != feature_names.size()) {
      return absl::InvalidArgumentError(
          "feature_names and output_feature_names must have the same size.");
    }
    absl::flat_hash_set<absl::string_view> unique_outputs;
    unique_outputs.reserve(output_feature_names.size());
    for (const auto& name : output_feature_names) {
      if (name.empty()) {
        return absl::InvalidArgumentError(
            "CpuEmbeddingLookupTransform requires non-empty output feature "
            "names.");
      }
      if (!unique_outputs.insert(name).second) {
        return absl::InvalidArgumentError(
            absl::StrCat("Duplicate output feature name found: ", name));
      }
    }
  }
  return std::unique_ptr<CpuEmbeddingLookupTransform>(
      new CpuEmbeddingLookupTransform(
          std::move(resource), std::move(feature_names),
          std::move(output_feature_names), dense_lookup, embedding_dim,
          std::move(output_shapes)));
}

std::vector<std::shared_ptr<neptune::data::Resource>>
CpuEmbeddingLookupTransform::GetAllResources() const {
  return {resource_};
}

// Static helper to validate lookup inputs.
absl::Status ValidateLookupInputs(const xla_serving::HostBuffer& ids,
                                  const xla_serving::HostBuffer& row_pointers,
                                  const xla_serving::HostBuffer& weights) {
  if (row_pointers.shape().dims().empty()) {
    return absl::InvalidArgumentError(
        "row_pointers HostBuffer must be at least 1D (got scalar).");
  }
  if (ids.DebugString() == "Empty host buffer") {
    return absl::InvalidArgumentError("ids HostBuffer is uninitialized.");
  }
  if (row_pointers.DebugString() == "Empty host buffer") {
    return absl::InvalidArgumentError(
        "row_pointers HostBuffer is uninitialized.");
  }
  if (weights.DebugString() == "Empty host buffer") {
    return absl::InvalidArgumentError("weights HostBuffer is uninitialized.");
  }

  if (ids.dtype().kind() != xla::ifrt::DType::Kind::kS64 &&
      ids.dtype().kind() != xla::ifrt::DType::Kind::kS32) {
    return absl::InvalidArgumentError("Embedding IDs must be int32 or int64.");
  }
  if (row_pointers.dtype().kind() != xla::ifrt::DType::Kind::kS32) {
    return absl::InvalidArgumentError("Row pointers must be int32 (kS32).");
  }
  if (weights.shape().num_elements() > 0 &&
      weights.dtype().kind() != xla::ifrt::DType::Kind::kF32) {
    return absl::InvalidArgumentError("Weights must be float32 (kF32).");
  }

  if (ids.shape().num_elements() > 0 && ids.base() == nullptr) {
    return absl::InvalidArgumentError("ids HostBuffer base pointer is null.");
  }
  if (row_pointers.shape().num_elements() > 0 &&
      row_pointers.base() == nullptr) {
    return absl::InvalidArgumentError(
        "row_pointers HostBuffer base pointer is null.");
  }
  if (weights.shape().num_elements() > 0 && weights.base() == nullptr) {
    return absl::InvalidArgumentError(
        "weights HostBuffer base pointer is null.");
  }

  auto rp_flat = row_pointers.flat<int32_t>();
  size_t rp_size = rp_flat.size();
  if (rp_size < 2) {
    return absl::InvalidArgumentError(
        "Row pointers must have at least 2 elements.");
  }
  const int32_t* rp_ptr = rp_flat.data();
  if (rp_ptr[0] != 0) {
    return absl::InvalidArgumentError(
        "First element of row pointers must be 0.");
  }

  size_t ids_size = 0;
  if (ids.dtype().kind() == xla::ifrt::DType::Kind::kS32) {
    ids_size = ids.flat<int32_t>().size();
  } else {
    ids_size = ids.flat<int64_t>().size();
  }

  if (rp_ptr[rp_size - 1] < 0) {
    return absl::InvalidArgumentError(
        "Last element of row pointers must be non-negative.");
  }

  if (static_cast<size_t>(rp_ptr[rp_size - 1]) != ids_size) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Embedding ids size (%d) does not match last element of row pointers "
        "(%d)",
        ids_size, rp_ptr[rp_size - 1]));
  }

  if (weights.shape().num_elements() > 0) {
    size_t weights_size = weights.flat<float>().size();
    if (weights_size != ids_size) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Weights size (%d) does not match expected size (%d)",
                          weights_size, ids_size));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<xla_serving::HostBuffer>>
CpuEmbeddingLookupTransform::operator()(
    std::vector<xla_serving::HostBuffer> ids, std::vector<xla_serving::HostBuffer> row_pointers,
    std::vector<xla_serving::HostBuffer> weights) {
  if (ids.size() != feature_names_.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "CpuEmbeddingLookupTransform expected ", feature_names_.size(),
        " ids buffers, but got ", ids.size(), "."));
  }
  if (!dense_lookup_) {
    if (row_pointers.size() != feature_names_.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "CpuEmbeddingLookupTransform expected ", feature_names_.size(),
          " row_pointers buffers, but got ", row_pointers.size(), "."));
    }
    if (weights.size() != feature_names_.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "CpuEmbeddingLookupTransform expected ", feature_names_.size(),
          " weights buffers, but got ", weights.size(), "."));
    }
  }

  std::vector<xla_serving::HostBuffer> outputs;
  outputs.reserve(feature_names_.size());

  for (size_t i = 0; i < feature_names_.size(); ++i) {
    const auto& feature_name = feature_names_[i];
    xla_serving::HostBuffer ids_hb = ids[i];

    if (ids_hb.DebugString() == "Empty host buffer" ||
        ids_hb.base() == nullptr) {
      return absl::InvalidArgumentError(
          "ids HostBuffer is uninitialized or null.");
    }

    const int32_t* ids_ptr_s32 = nullptr;
    const int64_t* ids_ptr_s64 = nullptr;
    std::vector<int64_t> temp_ids_s64;

    if (ids_hb.dtype().kind() == xla::ifrt::DType::Kind::kS32) {
      auto flat_ids = ids_hb.flat<int32_t>();
      ids_ptr_s32 = flat_ids.data();
      const int64_t num_ids = flat_ids.size();
      temp_ids_s64.resize(num_ids);
      for (size_t k = 0; k < num_ids; ++k) {
        temp_ids_s64[k] = ids_ptr_s32[k];
      }
      ids_ptr_s64 = temp_ids_s64.data();
    } else if (ids_hb.dtype().kind() == xla::ifrt::DType::Kind::kS64) {
      auto flat_ids = ids_hb.flat<int64_t>();
      ids_ptr_s64 = flat_ids.data();
    } else {
      return absl::InvalidArgumentError(
          "Embedding IDs must be int32 or int64.");
    }
    auto ids_span =
        absl::MakeConstSpan(ids_ptr_s64, ids_hb.shape().num_elements());

    int64_t batch_size = 0;
    std::vector<int32_t> temp_row_pointers;
    absl::Span<const int32_t> row_pointers_span;
    absl::Span<const float> weights_span;

    if (dense_lookup_) {
      if (ids_hb.shape().dims().size() != 2) {
        return absl::InvalidArgumentError(
            "Dense embedding IDs must be a 2D HostBuffer [batch_size, "
            "valency].");
      }
      batch_size = ids_hb.shape().dims()[0];
      int64_t valency = ids_hb.shape().dims()[1];
      if (batch_size < 0 || valency < 0) {
        return absl::InvalidArgumentError(
            "Dense embedding IDs dimensions must be non-negative.");
      }
      if (batch_size >= INT32_MAX) {
        return absl::InvalidArgumentError(
            "Batch size exceeds INT32_MAX bounds.");
      }
      if (batch_size > 0 && valency > INT32_MAX / batch_size) {
        return absl::InvalidArgumentError(
            "Batch volume (batch_size * valency) exceeds INT32_MAX bounds.");
      }
      temp_row_pointers.resize(batch_size + 1);
      for (int64_t b = 0; b <= batch_size; ++b) {
        temp_row_pointers[b] = b * valency;
      }
      row_pointers_span = absl::MakeConstSpan(temp_row_pointers);
    } else {
      xla_serving::HostBuffer row_pointers_hb = row_pointers[i];
      xla_serving::HostBuffer weights_hb = weights[i];

      RETURN_IF_ERROR(
          ValidateLookupInputs(ids_hb, row_pointers_hb, weights_hb));
      batch_size = row_pointers_hb.shape().num_elements() > 0
                       ? row_pointers_hb.shape().dims()[0] - 1
                       : 0;

      auto rp_flat = row_pointers_hb.flat<int32_t>();
      const int32_t* rp_ptr = rp_flat.size() > 0 ? rp_flat.data() : nullptr;

      auto w_flat = weights_hb.flat<float>();
      const float* w_ptr = w_flat.size() > 0 ? w_flat.data() : nullptr;

      row_pointers_span = absl::MakeConstSpan(rp_ptr, rp_flat.size());
      weights_span = absl::MakeConstSpan(w_ptr, w_flat.size());
    }

    xla_serving::HostBuffer output_hb;
    auto allocator = [&](size_t size) {
      int64_t dim1 = size;
      int64_t dim2 = 1;
      if (batch_size > 0 && size % batch_size == 0) {
        dim1 = batch_size;
        dim2 = size / batch_size;
        output_hb =
            xla_serving::HostBuffer(xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
                            xla::ifrt::Shape({dim1, dim2}));
      } else {
        output_hb =
            xla_serving::HostBuffer(xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
                            xla::ifrt::Shape({static_cast<int64_t>(size)}));
      }
      auto out_flat = output_hb.flat<float>();
      float* data = out_flat.data();
      return absl::MakeSpan(data, size);
    };

    RETURN_IF_ERROR(resource_->Get()
                        ->Execute<float>(feature_name, ids_span,
                                         row_pointers_span, weights_span,
                                         std::move(allocator))
                        .status());

    outputs.push_back(std::move(output_hb));
  }

  return outputs;
}

absl::Status CpuEmbeddingLookupTransform::PropagateSignatureImpl(
    api::ShapeInferenceContext& context) const {
  if (context.inputs.empty() || !context.inputs[0].IsSpecList()) {
    return absl::InvalidArgumentError(
        "CpuEmbeddingLookupTransform expected a list of tensors as input for "
        "IDs.");
  }
  const auto& ids_specs = context.inputs[0].spec_list();
  std::vector<api::HostBufferSpec> output_specs;
  output_specs.reserve(feature_names_.size());

  constexpr int64_t kDynamicDimension = -1;
  for (size_t i = 0; i < feature_names_.size(); ++i) {
    std::vector<int64_t> output_dims;

    if (i < output_shapes_.size() && !output_shapes_[i].empty()) {
      output_dims.reserve(output_shapes_[i].size());
      for (int64_t d : output_shapes_[i]) {
        output_dims.push_back(d < 0 ? kDynamicDimension : d);
      }
    } else {
      int64_t embedding_dim = embedding_dim_;
      output_dims.push_back(kDynamicDimension);

      if (i < ids_specs.size()) {
        absl::Span<const int64_t> id_dims = ids_specs[i].shape.dims();
        if (id_dims.size() == 3 || (id_dims.size() == 2 && !dense_lookup_)) {
          output_dims.push_back(id_dims[1]);
        }
      }
      output_dims.push_back(embedding_dim);
    }

    output_specs.push_back(api::HostBufferSpec(
        xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
        xla::ifrt::Shape(absl::MakeConstSpan(output_dims))));
  }

  context.outputs.push_back(api::ShapeInferenceValue(std::move(output_specs)));
  return absl::OkStatus();
}

absl::StatusOr<TransformConfig> CpuEmbeddingLookupTransform::ToProto() const {
  TransformConfig config;
  config.set_op_name("CpuEmbeddingLookupTransform");
  config.set_name(feature_names_.empty() ? "" : feature_names_[0]);
  if (resource_ != nullptr) {
    config.add_resource_ids(resource_->id());
  }

  namespace jce = ::neptune::data::composite::transforms::cpu_embedding;
  jce::CpuEmbeddingLookupTransformConfig detail;
  if (!feature_names_.empty()) {
    detail.set_feature(feature_names_[0]);
    for (const auto& name : feature_names_) {
      detail.add_feature_names(name);
    }
  }
  for (const auto& name : output_feature_names_) {
    detail.add_output_feature_names(name);
  }
  detail.set_dense_lookup(dense_lookup_);
  detail.set_embedding_dim(embedding_dim_);
  config.mutable_detail()->PackFrom(detail);

  return config;
}

ND_REGISTER_TRANSFORM(CpuEmbeddingLookupTransform,
                      "CpuEmbeddingLookupTransform")
    .SetDeserializer(
        [](const TransformConfig& config,
           absl::Span<const std::shared_ptr<neptune::data::Resource>> resources)
            -> absl::StatusOr<std::unique_ptr<TransformBase>> {
          if (resources.empty()) {
            return absl::InvalidArgumentError(
                "CpuEmbeddingLookupTransform requires a resource.");
          }
          auto resource = std::dynamic_pointer_cast<CpuEmbeddingEngineResource>(
              const_cast<std::shared_ptr<neptune::data::Resource>&>(
                  resources[0]));
          if (resource == nullptr) {
            return absl::InvalidArgumentError(
                "Resource is not a CpuEmbeddingEngineResource.");
          }

          namespace jce = ::neptune::data::composite::transforms::cpu_embedding;
          jce::CpuEmbeddingLookupTransformConfig detail;
          std::vector<std::string> feature_names;
          std::vector<std::string> output_feature_names;
          bool dense_lookup = false;
          int64_t embedding_dim = 128;
          if (config.has_detail() && config.detail().UnpackTo(&detail)) {
            if (detail.feature_names_size() > 0) {
              feature_names.reserve(detail.feature_names_size());
              for (const auto& name : detail.feature_names()) {
                feature_names.push_back(std::string(name));
              }
            } else if (!detail.feature().empty()) {
              feature_names.push_back(std::string(detail.feature()));
            }
            if (detail.output_feature_names_size() > 0) {
              output_feature_names.reserve(detail.output_feature_names_size());
              for (const auto& name : detail.output_feature_names()) {
                output_feature_names.push_back(std::string(name));
              }
            }
            dense_lookup = detail.dense_lookup();
            if (detail.embedding_dim() > 0) {
              embedding_dim = detail.embedding_dim();
            }
          }
          if (feature_names.empty()) {
            feature_names.push_back(std::string(config.name()));
          }

          return CpuEmbeddingLookupTransform::Create(
              std::move(resource), std::move(feature_names), dense_lookup,
              std::move(output_feature_names), embedding_dim);
        });

}  // namespace composite
}  // namespace neptune::data
