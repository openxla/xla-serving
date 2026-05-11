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

#include "runtime/interpreter/manifest_utils.h"  // from @neptune

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/ascii.h"  // from @abseil-cpp
#include "absl/strings/match.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_format.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp
#include "absl/strings/str_split.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/strings/strip.h"  // from @abseil-cpp
#include "absl/strings/substitute.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "protos/manifest.pb.h"  // from @neptune
#include "protos/xla/pjrt/proto/compile_options.pb.h"  // from @neptune
#include "protos/xla/xla_data.pb.h"  // from @neptune
#include "protos/orchestration.pb.h"  // from @neptune
#include "protos/type.pb.h"  // from @neptune
#include "runtime/core/constants.h"  // from @neptune
#include "runtime/core/flat_signature.h"  // from @neptune
#include "runtime/core/stablehlo_utils.h"  // from @neptune
#include "runtime/core/type.h"  // from @neptune
#include "runtime/interpreter/orbax_type_utils.h"  // from @neptune
#include "runtime/kernel/kernel_builder_constants.h"  // from @neptune
#include "google/protobuf/descriptor.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "google/protobuf/util/message_differencer.h"
#include "xla/debug_options_flags.h"  // from @xla
#include "xla/hlo/ir/hlo_sharding.h"  // from @xla
#include "xla/layout.h"  // from @xla
#include "xla/pjrt/proto/compile_options.pb.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla
#include "tsl/platform/path.h"  // from @tsl

namespace neptune {

using type::Dict;
using type::None;
using type::StringTypePairs;
using type::Tuple;
using type::Type;

namespace {

template <typename To, typename From>
absl::StatusOr<To> ConvertProto(const From& from) {
  To to;
  if (!to.ParseFromString(from.SerializeAsString())) {
    return absl::InternalError("Failed to convert proto via serialization.");
  }
  return to;
}

// Orbax model version metadata (internal representation).
struct ModelVersionMetadata {
  std::string version;
  std::string mime_type;
  std::string manifest_file_path;
};

// Converts a vector of `neptune_model_type::LeafType` to a vector of `ArgSpec`.
// We assume that the leaf types are all `TensorType`.
absl::StatusOr<std::vector<ArgSpec>> ConvertToArgSpecs(
    absl::Span<const neptune_model_type::LeafType> flatten_types) {
  std::vector<ArgSpec> input_specs;
  // TODO: populate shape from function body
  input_specs.reserve(flatten_types.size());
  for (int i = 0; i < flatten_types.size(); ++i) {
    TF_ASSIGN_OR_RETURN(ArgSpec spec,
                        ConvertToArgSpec(flatten_types[i].tensor_type()));
    input_specs.push_back(std::move(spec));
  }
  return input_specs;
}

// Gets the orchestration data from the manifest.
absl::StatusOr<const neptune_model_manifest::UnstructuredData> GetOrchestration(
    const neptune_model_manifest::Manifest& manifest) {
  const auto& dict = manifest.supplemental_info();
  auto it = dict.find(kOrchestrationSupplementalName);
  if (it == dict.end()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Manifest does not have a global supplemental with name \"",
        kOrchestrationSupplementalName, "\"."));
  }
  return it->second;
}

// Gets the name of the data case.
std::string GetDataCaseName(
    neptune_model_manifest::UnstructuredData::DataCase data_case) {
  const google::protobuf::FieldDescriptor* field_descriptor =
      neptune_model_manifest::UnstructuredData::descriptor()->FindFieldByNumber(
          data_case);
  return field_descriptor == nullptr ? "DATA_NOT_SET"
                                     : std::string(field_descriptor->name());
}

template <typename Dict>
std::vector<std::pair<std::string, typename Dict::mapped_type>>
SortedByStringKey(const Dict& dict) {
  std::vector<std::pair<std::string, typename Dict::mapped_type>> pairs;
  for (const auto& [key, value] : dict) {
    pairs.push_back(std::make_pair(key, value));
  }
  std::sort(pairs.begin(), pairs.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  return pairs;
}

bool IsSequence(const neptune_model_type::Type& type) {
  return type.has_tuple() || type.has_list();
}

// Returns the elements of a sequence type. Assumes that the type is a sequence.
const google::protobuf::RepeatedPtrField<neptune_model_type::Type>&
GetElementsFromSequence(const neptune_model_type::Type& type) {
  return type.has_tuple() ? type.tuple().elements() : type.list().elements();
}

// Returns true if the `type` matches the args-kwargs ([(), {}]) pattern.
bool IsArgsKwargsPattern(const neptune_model_type::Type& type) {
  // TODO: consider using a tree structure (e.g. `neptune::type::Type`)
  // to first translate `neptune_model_type::Type` to this tree structure to
  // simplify the logic.
  if (!IsSequence(type)) {
    return false;
  }
  const google::protobuf::RepeatedPtrField<neptune_model_type::Type>& elements =
      GetElementsFromSequence(type);
  if (elements.size() != 2) {
    return false;
  }
  return IsSequence(elements[0]) && elements[1].has_dict();
}

// Returns true if the `type` matches the args-hybrids-kwargs ([(), {}/(), {}])
// pattern.
bool IsArgsHybridsKwargsPattern(const neptune_model_type::Type& type) {
  if (!IsSequence(type)) {
    return false;
  }
  const google::protobuf::RepeatedPtrField<neptune_model_type::Type>& elements =
      GetElementsFromSequence(type);
  if (elements.size() != 3) {
    return false;
  }
  return IsSequence(elements[0]) && elements[1].has_string_type_pairs() &&
         elements[2].has_dict();
}

// Fills `specs` from `type`. The `name` is used to construct the name in the
// `MetaArgSpec`.
absl::Status FillMetaArgSpecs(const neptune_model_type::Type& type,
                              std::vector<MetaArgSpec>& specs) {
  if (type.has_leaf()) {
    TF_ASSIGN_OR_RETURN(ArgSpec spec,
                        ConvertToArgSpec(type.leaf().tensor_type()));
    specs.push_back(MetaArgSpec{
        .spec = std::move(spec),
    });
    return absl::OkStatus();
  }

  switch (type.type_case()) {
    case neptune_model_type::Type::kTuple:
      [[fallthrough]];
    case neptune_model_type::Type::kList:
      for (const auto& element : GetElementsFromSequence(type)) {
        TF_RETURN_IF_ERROR(FillMetaArgSpecs(element, specs));
      }
      return absl::OkStatus();
    case neptune_model_type::Type::kDict:
      // We sort by the string key, aligned with JAX and OBM Python.
      for (const auto& [key, value] :
           SortedByStringKey(type.dict().string_to_type())) {
        TF_RETURN_IF_ERROR(FillMetaArgSpecs(value, specs));
      }
      return absl::OkStatus();
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported type in `Type`: ", type));
  }
}

// Infers the assigned devices from the function signature.
// This is limited to just single host SPMD.
// Ideally, we should get the device assignment from the manifest directly,
// but it is not always available in the manifest. This is a best effort
// implementation to avoid hard code the device assignment.
absl::StatusOr<AssignedDevices> InferAssignedDevicesFromFunctionSignature(
    const neptune_model_manifest::Function& stable_hlo_func) {
  int64_t max_num_devices = 0;
  std::vector<neptune_model_type::LeafType> leafs;
  TF_RETURN_IF_ERROR(FlattenType(stable_hlo_func.signature().input(), leafs));

  for (const auto& leaf : leafs) {
    TF_ASSIGN_OR_RETURN(
        xla::OpSharding xla_sharding,
        ConvertProto<xla::OpSharding>(leaf.tensor_type().sharding()));
    TF_ASSIGN_OR_RETURN(xla::HloSharding hlo_sharding,
                        xla::HloSharding::FromProto(xla_sharding));
    VLOG(2) << "sharding: " << xla_sharding << " have "
            << hlo_sharding.TotalNumTiles() << " tiles.";
    max_num_devices = std::max(max_num_devices, hlo_sharding.TotalNumTiles());
  }

  if (max_num_devices <= 0) {
    return absl::InvalidArgumentError(
        "Failed to infer number of cores per replica from function signature");
  }
  VLOG(2) << "max_num_devices: " << max_num_devices;
  return AssignedDevices{
      .num_replicas = 1,
      .num_cores_per_replica = static_cast<int>(max_num_devices),
  };
}

absl::StatusOr<ModelVersionMetadata> ParseModelVersionFile(
    absl::string_view model_path) {
  std::string metadata_str;
  auto status = tsl::ReadFileToString(
      tsl::Env::Default(),
      tsl::io::JoinPath(model_path, kNeptuneModelManifestVersionTxt),
      &metadata_str);
  if (!status.ok()) {
    TF_RETURN_IF_ERROR(tsl::ReadFileToString(
        tsl::Env::Default(),
        tsl::io::JoinPath(model_path, kOrbaxModelManifestVersionTxt),
        &metadata_str));
  }

  std::string version;
  std::string mime_type;
  std::string manifest_file_path;

  std::vector<absl::string_view> lines = absl::StrSplit(metadata_str, '\n');
  for (const auto line : lines) {
    auto stripped_line = absl::StripAsciiWhitespace(line);
    if (stripped_line.empty()) {
      continue;
    }
    std::vector<absl::string_view> parts =
        absl::StrSplit(stripped_line, absl::MaxSplits(':', 1));
    if (parts.size() != 2) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid line in model version metadata: ", line));
    }
    auto key = absl::StripAsciiWhitespace(parts[0]);
    auto value = absl::StripAsciiWhitespace(parts[1]);

    if (!absl::StartsWith(value, "\"") || !absl::EndsWith(value, "\"")) {
      return absl::InvalidArgumentError(
          absl::StrCat("Value is not double-quoted: ", value));
    }

    value = absl::StripPrefix(absl::StripSuffix(value, "\""), "\"");

    // Duplicate keys are overwritten.
    if (key == kOrbaxModelManifestVersionKey) {
      version = value;
    } else if (key == kOrbaxModelManifestMimeTypeKey) {
      mime_type = value;
    } else if (key == kOrbaxModelManifestManifestFilePathKey) {
      manifest_file_path = value;
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Unknown key in model version metadata: ", key));
    }
  }

  return ModelVersionMetadata{.version = version,
                              .mime_type = mime_type,
                              .manifest_file_path = manifest_file_path};
}

}  // namespace

absl::StatusOr<ArgSpec> ConvertToArgSpec(
    const neptune_model_type::TensorType& type) {
  TF_ASSIGN_OR_RETURN(xla::ifrt::DType dtype, ToIfrtDType(type.dtype()));
  TF_ASSIGN_OR_RETURN(ArgShape shape, ToArgShape(type.shape()));
  TF_ASSIGN_OR_RETURN(xla::OpSharding xla_sharding,
                      ConvertProto<xla::OpSharding>(type.sharding()));
  TF_ASSIGN_OR_RETURN(xla::HloSharding hlo_sharding,
                      xla::HloSharding::FromProto(xla_sharding));
  std::optional<xla::Layout> layout = std::nullopt;
  if (type.has_layout()) {
    TF_ASSIGN_OR_RETURN(xla::LayoutProto xla_layout,
                        ConvertProto<xla::LayoutProto>(type.layout()));
    TF_ASSIGN_OR_RETURN(layout, xla::Layout::FromProto(xla_layout));
  }
  return ArgSpec{
      .dtype = std::move(dtype),
      .shape = std::move(shape),
      .sharding = std::move(hlo_sharding),
      .layout = std::move(layout),
  };
}

absl::StatusOr<xla::ifrt::DType> ToIfrtDType(neptune_model_type::DType type) {
  if (type == neptune_model_type::DType::str) {
    // This type has no corresponding `xla::PrimitiveType`.
    return xla::ifrt::DType(xla::ifrt::DType::kString);
  }
  TF_ASSIGN_OR_RETURN(xla::PrimitiveType primitive_type, ToPrimitiveType(type));
  return xla::ifrt::ToDType(primitive_type);
}

absl::StatusOr<ArgShape> ToArgShape(const neptune_model_type::Shape& shape) {
  if (shape.shape_with_known_rank().dimension_sizes().empty()) {
    return ArgShape({});
  }
  const google::protobuf::RepeatedPtrField<neptune_model_type::DimensionSize>&
      dimension_sizes = shape.shape_with_known_rank().dimension_sizes();

  ArgShape arg_shape;
  arg_shape.reserve(dimension_sizes.size());
  for (const auto& dim : dimension_sizes) {
    if (dim.has_size()) {
      arg_shape.push_back(dim.size());
    } else {
      arg_shape.push_back(std::nullopt);
    }
  }
  return arg_shape;
}

// TODO(wangpeng): flatten_types's type can directly be
//   vector<neptune_model_manifest::TensorType>.
absl::Status FlattenType(
    const neptune_model_type::Type& type,
    std::vector<neptune_model_type::LeafType>& flatten_types) {
  switch (type.type_case()) {
    case neptune_model_type::Type::kLeaf: {
      if (!type.leaf().has_tensor_type()) {
        return absl::UnimplementedError(absl::StrCat(
            "Unsupported leaf type: ", type.leaf().SerializeAsString()));
      }
      flatten_types.push_back(type.leaf());
      return absl::OkStatus();
    }
    case neptune_model_type::Type::kNone:
      // We can just ignore the none case.
      return absl::OkStatus();
    case neptune_model_type::Type::kTuple:
      [[fallthrough]];
    case neptune_model_type::Type::kList: {
      for (const auto& element : (type.has_tuple() ? type.tuple().elements()
                                                   : type.list().elements())) {
        TF_RETURN_IF_ERROR(FlattenType(element, flatten_types));
      }
      return absl::OkStatus();
    }
    case neptune_model_type::Type::kDict: {
      // We sort by the string key, aligned with JAX and OBM Python.
      for (const auto& [_, value] :
           SortedByStringKey(type.dict().string_to_type())) {
        TF_RETURN_IF_ERROR(FlattenType(value, flatten_types));
      }
      return absl::OkStatus();
    }
    case neptune_model_type::Type::kStringTypePairs: {
      for (const auto& element : type.string_type_pairs().elements()) {
        TF_RETURN_IF_ERROR(FlattenType(element.snd(), flatten_types));
      }
      return absl::OkStatus();
    }
    case neptune_model_type::Type::TYPE_NOT_SET:
      // Orbax Model should not allow this case.
      return absl::InvalidArgumentError(
          "The oneof field `Type.type` is not set.");
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unsupported type: ", type.SerializeAsString()));
  }
}

absl::StatusOr<std::vector<ArgSpec>> GetInputSpec(
    const neptune_model_manifest::Function& func) {
  std::vector<neptune_model_type::LeafType> flatten_types;
  TF_RETURN_IF_ERROR(FlattenType(func.signature().input(), flatten_types));
  return ConvertToArgSpecs(flatten_types);
}

absl::StatusOr<std::vector<ArgSpec>> GetOutputSpec(
    const neptune_model_manifest::Function& func) {
  std::vector<neptune_model_type::LeafType> flatten_types;
  TF_RETURN_IF_ERROR(FlattenType(func.signature().output(), flatten_types));

  return ConvertToArgSpecs(flatten_types);
}

absl::Status CheckMimeType(
    const neptune_model_manifest::UnstructuredData& unstructured_data,
    absl::Span<const absl::string_view> allowed_mime_types,
    absl::string_view description) {
  if (!unstructured_data.has_mime_type()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s: mime type is not set.", description));
  }
  for (const auto& mime_type : allowed_mime_types) {
    if (unstructured_data.mime_type() == mime_type) {
      return absl::OkStatus();
    }
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "%s: only mime type \"%s\" is supported, but got mime type "
      "\"%s\"",
      description, absl::StrJoin(allowed_mime_types, "\" or \""),
      unstructured_data.mime_type()));
}

absl::Status CheckDataType(
    const neptune_model_manifest::UnstructuredData& unstructured_data,
    absl::Span<const neptune_model_manifest::UnstructuredData::DataCase>
        allowed_data_types,
    absl::string_view description) {
  for (const auto& allowed_data_type : allowed_data_types) {
    if (unstructured_data.data_case() == allowed_data_type) {
      return absl::OkStatus();
    }
  }
  std::vector<std::string> allowed_data_type_names;
  for (const auto& allowed_data_type : allowed_data_types) {
    allowed_data_type_names.push_back(GetDataCaseName(allowed_data_type));
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "%s: only data type \"%s\" is supported, but got data type \"%s\".",
      description, absl::StrJoin(allowed_data_type_names, "\" or \""),
      GetDataCaseName(unstructured_data.data_case())));
}

absl::Status CheckMimeAndDataType(
    const neptune_model_manifest::UnstructuredData& unstructured_data,
    absl::Span<const absl::string_view> allowed_mime_types,
    absl::Span<const neptune_model_manifest::UnstructuredData::DataCase>
        allowed_data_types,
    absl::string_view description) {
  TF_RETURN_IF_ERROR(
      CheckMimeType(unstructured_data, allowed_mime_types, description));
  return CheckDataType(unstructured_data, allowed_data_types, description);
}

absl::StatusOr<neptune_model_manifest::Manifest> GetManifest(
    absl::string_view model_path) {
  TF_ASSIGN_OR_RETURN(auto model_version_metadata,
                      ParseModelVersionFile(model_path));
  // TODO: Once there is more than one version of the model
  // we will need to check the list of supported versions and customize the
  // loading process accordingly.
  if (model_version_metadata.version != kOrbaxModelManifestVersion) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Unsupported version: ", model_version_metadata.version));
  }
  if (model_version_metadata.mime_type != kOrbaxModelManifestMimeType &&
      model_version_metadata.mime_type != kOrbaxModelManifestMimeTypeLegacy) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Unsupported type: ", model_version_metadata.mime_type));
  }
  if (model_version_metadata.manifest_file_path.empty()) {
    return absl::InvalidArgumentError("Manifest file path is empty");
  }
  auto manifest_path =
      tsl::io::JoinPath(model_path, model_version_metadata.manifest_file_path);
  neptune_model_manifest::Manifest manifest;
  TF_RETURN_IF_ERROR(
      tsl::ReadBinaryProto(tsl::Env::Default(), manifest_path, &manifest));
  return manifest;
}

absl::StatusOr<std::string> GetAbsoluteFileSystemPath(
    absl::string_view model_path,
    const neptune_model_manifest::FileSystemLocation& file_system_location) {
  if (!file_system_location.has_string_path()) {
    return absl::UnimplementedError(
        "FileSystemLocation does not have a string path. Only support string "
        "path for now.");
  }
  if (file_system_location.string_path().empty()) {
    return absl::InvalidArgumentError("Empty string path");
  }
  if (tsl::io::IsAbsolutePath(file_system_location.string_path())) {
    // This support can be added if needed.
    return absl::UnimplementedError(
        "Absolute path is not supported in `FileSystemLocation`.");
  }
  return tsl::io::JoinPath(model_path, file_system_location.string_path());
}

absl::StatusOr<FlatSignature> GetInputFlatSignature(
    const neptune_model_type::Type& type, int offset) {
  // Input type will have the args-kwargs pattern or
  // args-hybrids-kwargs pattern.
  bool is_args_kwargs_pattern = IsArgsKwargsPattern(type);
  bool is_args_hybrids_kwargs_pattern = IsArgsHybridsKwargsPattern(type);
  if (!is_args_kwargs_pattern && !is_args_hybrids_kwargs_pattern) {
    return absl::InvalidArgumentError(
        "Only args-kwargs pattern or args-hybrids-kwargs pattern is "
        "supported for input.");
  }

  const neptune_model_type::Type& positional_args =
      GetElementsFromSequence(type)[0];
  const neptune_model_type::Dict& kwargs =
      is_args_kwargs_pattern ? GetElementsFromSequence(type)[1].dict()
                             : GetElementsFromSequence(type)[2].dict();

  if (GetElementsFromSequence(positional_args).size() < offset) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Offset ", offset, " is out of range. Should be less than or equal to ",
        GetElementsFromSequence(positional_args).size()));
  }

  std::vector<MetaArgSpec> specs;
  for (int i = offset; i < GetElementsFromSequence(positional_args).size();
       ++i) {
    TF_RETURN_IF_ERROR(
        FillMetaArgSpecs(GetElementsFromSequence(positional_args)[i], specs));
  }
  if (is_args_hybrids_kwargs_pattern) {
    for (const auto& element :
         GetElementsFromSequence(type)[1].string_type_pairs().elements()) {
      TF_RETURN_IF_ERROR(FillMetaArgSpecs(element.snd(), specs));
    }
  }
  for (const auto& [key, value] : SortedByStringKey(kwargs.string_to_type())) {
    TF_RETURN_IF_ERROR(FillMetaArgSpecs(value, specs));
  }
  return FlatSignature(std::move(specs));
}

absl::StatusOr<FlatSignature> GetOutputFlatSignature(
    const neptune_model_type::Type& type) {
  std::vector<MetaArgSpec> specs;
  TF_RETURN_IF_ERROR(FillMetaArgSpecs(type, specs));
  return FlatSignature(std::move(specs));
}

absl::StatusOr<AssignedDevices> GetAssignedDevices(
    const neptune_model_manifest::Manifest& manifest,
    const neptune_model_manifest::Function& stable_hlo_func,
    const xla::CompileOptionsProto* absl_nullable compile_options,
    bool ignore_device_assignment_in_manifest) {
  // If the device assignment is not available in the manifest, we infer it
  // from the function signature.
  if (!manifest.has_device_assignment_by_coords()) {
    return InferAssignedDevicesFromFunctionSignature(stable_hlo_func);
  }

  if (compile_options == nullptr) {
    return absl::InvalidArgumentError(
        "Compile options is required to get the assigned devices.");
  }

  LOG(INFO) << "Using compile options from model: " << *compile_options;
  // Only support single device for now.
  int num_replicas = compile_options->executable_build_options().num_replicas();
  // Defaults to 1 unless the device assignment by coords is set.
  int num_cores_per_replica =
      compile_options->executable_build_options().num_partitions();

  // Get the device assignment from the device assignment by coords.
  // Device assignment by coords is a list of [x,y,z,core_on_chip] for each
  // device.
  int num_devices = 0;
  std::vector<int> device_assignment;
  if (manifest.has_device_assignment_by_coords()) {
    for (const auto& device :
         manifest.device_assignment_by_coords().devices()) {
      // `coords` and `core_on_chip` should be both set or empty.
      if (device.coords().empty() == device.has_core_on_chip()) {
        return absl::InvalidArgumentError(absl::Substitute(
            "Device assignment by coords should either have both coords "
            "and core_on_chip set or empty, but got $0",
            device.DebugString()));
      }
      for (const auto& coord : device.coords()) {
        device_assignment.push_back(coord);
      }
      if (device.has_core_on_chip()) {
        device_assignment.push_back(device.core_on_chip());
      }
      num_devices++;
    }
  }

  VLOG(2) << "num_replicas: " << num_replicas
          << " num_cores_per_replica: " << num_cores_per_replica
          << " num_devices: " << num_devices;

  if (num_devices != num_replicas * num_cores_per_replica) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Found ", num_devices,
        " devices, which does not match the number of replicas: ", num_replicas,
        " and cores per replica: ", num_cores_per_replica));
  }
  if (ignore_device_assignment_in_manifest || device_assignment.empty()) {
    // No explicit device assignment.
    return AssignedDevices{
        .num_replicas = num_replicas,
        .num_cores_per_replica = num_cores_per_replica,
    };
  } else {
    num_cores_per_replica = num_devices;
    return AssignedDevices{
        .device_assignment = std::move(device_assignment),
        .num_replicas = num_replicas,
        .num_cores_per_replica = num_cores_per_replica,
    };
  }
}

absl::StatusOr<FlatSignature> GetFlatSignatureFromNamedSignature(
    const neptune_model_orchestration::NamedTensorTypeList& inputs_or_outputs) {
  if (inputs_or_outputs.named_tensor_types().empty()) {
    return absl::InvalidArgumentError(
        "Empty named signature. Not able to get the flat signature.");
  }
  std::vector<MetaArgSpec> specs;
  specs.reserve(inputs_or_outputs.named_tensor_types_size());
  for (const auto& element : inputs_or_outputs.named_tensor_types()) {
    TF_ASSIGN_OR_RETURN(ArgSpec spec, ConvertToArgSpec(element.tensor_type()));
    if (element.name().empty()) {
      return absl::InvalidArgumentError("Empty name in named signature.");
    }
    specs.push_back(MetaArgSpec{
        .spec = std::move(spec),
        .name = element.name(),
    });
  }
  return FlatSignature(std::move(specs));
}

absl::StatusOr<std::optional<xla::CompileOptionsProto>> GetCompileOptions(
    absl::Span<const neptune_model_manifest::Function* const> stablehlo_funcs,
    absl::string_view platform_name) {
  std::optional<xla::CompileOptionsProto> compile_options;
  for (const auto& func : stablehlo_funcs) {
    const neptune_model_manifest::StableHloFunctionBody& stable_hlo_body =
        func->body().stable_hlo_body();
    auto xla_options_it =
        stable_hlo_body.supplemental_info().find(kXlaCompileOptions);
    // TODO: Returns error here once the default compile options
    // are supported.
    if (xla_options_it != stable_hlo_body.supplemental_info().end()) {
      TF_RETURN_IF_ERROR(CheckMimeAndDataType(
          xla_options_it->second,
          {kXlaCompileOptionsMimeType, kXlaCompileOptionsMimeTypeLegacy},
          {neptune_model_manifest::UnstructuredData::kInlinedBytes},
          "xla_compile_options supplemental"));
      std::optional<neptune_model_manifest::CompileOptionsProtoMap>
          compile_options_map;
      compile_options_map.emplace();
      if (!compile_options_map->ParseFromString(
              xla_options_it->second.inlined_bytes())) {
        return absl::InvalidArgumentError(
            "Failed to parse the global supplemental with name "
            "'xla_compile_options'.");
      }
      auto compile_options_it =
          compile_options_map->map().find(absl::AsciiStrToLower(platform_name));
      if (compile_options_it != compile_options_map->map().end()) {
        TF_ASSIGN_OR_RETURN(
            xla::CompileOptionsProto xla_opts,
            ConvertProto<xla::CompileOptionsProto>(compile_options_it->second));
        if (!compile_options.has_value()) {
          compile_options = xla_opts;
        } else if (!google::protobuf::util::MessageDifferencer::Equivalent(
                       compile_options.value(), xla_opts)) {
          // Ensure that the compile options are consistent across all
          // functions.
          return absl::InvalidArgumentError(
              "Compilation options are not consistent across all functions.");
        }
      }
    }
  }

  // Default initializing debug options if not initialized during export.
  // TODO (b/510835561) Remove override of debug options once set in export
  if (compile_options.has_value() &&
      compile_options->has_executable_build_options() &&
      !compile_options->executable_build_options().has_debug_options()) {
    *compile_options->mutable_executable_build_options()
         ->mutable_debug_options() = xla::GetDebugOptionsFromFlags();
  }

  return compile_options;
}

absl::StatusOr<std::vector<std::string>> GetInputNames(
    absl::string_view orchestration_path, absl::string_view pipeline_name) {
  // Get the pipeline.
  neptune_model_orchestration::Pipelines pipelines;
  TF_RETURN_IF_ERROR(tsl::ReadBinaryProto(
      tsl::Env::Default(), std::string(orchestration_path), &pipelines));
  const auto& pipeline_map = pipelines.name_to_pipeline();
  auto it = pipeline_map.find(std::string(pipeline_name));
  if (it == pipeline_map.end()) {
    std::vector<std::string> pipeline_names;
    pipeline_names.reserve(pipeline_map.size());
    for (const auto& pair : pipeline_map) {
      pipeline_names.push_back(pair.first);
    }
    return absl::InvalidArgumentError(absl::StrCat(
        "Pipeline \"", pipeline_name, "\" not found. Available pipelines: ",
        absl::StrJoin(pipeline_names, ", ")));
  }
  const neptune_model_orchestration::Pipeline& pipeline = it->second;
  // Get the input flat signature from the pipeline.
  if (!pipeline.has_named_signature()) {
    return absl::InvalidArgumentError(
        "Pipeline does not have a named signature.");
  }
  if (!pipeline.named_signature().has_inputs()) {
    return absl::InvalidArgumentError("Named signature does not have inputs.");
  }
  TF_ASSIGN_OR_RETURN(
      const auto input_flat_signature,
      GetFlatSignatureFromNamedSignature(pipeline.named_signature().inputs()));
  // Get all the input names.
  TF_ASSIGN_OR_RETURN(auto names, input_flat_signature.GetAllNames());
  return std::vector<std::string>(names.begin(), names.end());
}

absl::StatusOr<std::vector<std::string>> GetOutputNames(
    absl::string_view orchestration_path, absl::string_view pipeline_name) {
  // Get the pipeline.
  neptune_model_orchestration::Pipelines pipelines;
  TF_RETURN_IF_ERROR(tsl::ReadBinaryProto(
      tsl::Env::Default(), std::string(orchestration_path), &pipelines));
  const auto& pipeline_map = pipelines.name_to_pipeline();
  auto it = pipeline_map.find(std::string(pipeline_name));
  if (it == pipeline_map.end()) {
    std::vector<std::string> pipeline_names;
    pipeline_names.reserve(pipeline_map.size());
    for (const auto& pair : pipeline_map) {
      pipeline_names.push_back(pair.first);
    }
    return absl::InvalidArgumentError(absl::StrCat(
        "Pipeline \"", pipeline_name, "\" not found. Available pipelines: ",
        absl::StrJoin(pipeline_names, ", ")));
  }
  const neptune_model_orchestration::Pipeline& pipeline = it->second;
  // Get the output flat signature from the pipeline.
  if (!pipeline.has_named_signature()) {
    return absl::InvalidArgumentError(
        "Pipeline does not have a named signature.");
  }
  if (!pipeline.named_signature().has_outputs()) {
    return absl::InvalidArgumentError("Named signature does not have outputs.");
  }
  TF_ASSIGN_OR_RETURN(
      const auto output_flat_signature,
      GetFlatSignatureFromNamedSignature(pipeline.named_signature().outputs()));
  // Get all the output names.
  TF_ASSIGN_OR_RETURN(auto names, output_flat_signature.GetAllNames());
  return std::vector<std::string>(names.begin(), names.end());
}

absl::StatusOr<neptune_model_orchestration::Pipelines> GetOexOrchestration(
    absl::string_view model_path,
    const neptune_model_manifest::Manifest& manifest) {
  TF_ASSIGN_OR_RETURN(
      const neptune_model_manifest::UnstructuredData orchestration,
      GetOrchestration(manifest));

  TF_RETURN_IF_ERROR(CheckMimeAndDataType(
      orchestration,
      {kOexOrchestrationMultiPipelineMimeType,
       kOexOrchestrationMultiPipelineMimeTypeLegacy},
      {neptune_model_manifest::UnstructuredData::kFileSystemLocation},
      "Orchestration"));

  TF_ASSIGN_OR_RETURN(std::string orchestration_path,
                      GetAbsoluteFileSystemPath(
                          model_path, orchestration.file_system_location()));
  neptune_model_orchestration::Pipelines pipelines;

  TF_RETURN_IF_ERROR(tsl::ReadBinaryProto(tsl::Env::Default(),
                                          orchestration_path, &pipelines));
  return pipelines;
}

absl::StatusOr<std::vector<const neptune_model_manifest::Function*>>
GetStableHloFunctionsByObjectName(

    const absl::flat_hash_map<std::string,
                              const neptune_model_manifest::TopLevelObject*>&
        manifest_objects,
    absl::string_view function_object_name) {
  if (!manifest_objects.contains(function_object_name)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to find function object with name: \"",
                     function_object_name, "\""));
  }
  const neptune_model_manifest::TopLevelObject* function_obj =
      manifest_objects.at(function_object_name);

  std::vector<const neptune_model_manifest::Function*> functions;
  if (function_obj->has_function()) {
    // This is the case for
    // - a StableHLO with concrete input and output shapes.
    // - or a polymorphic StableHLO with symbolic input and output shapes.
    functions.push_back(&function_obj->function());
  } else if (function_obj->has_poly_fn()) {
    // This is the case for
    // - a set of stablehlos with specific input/output shapes that are
    //   concretized from a JAX polymorphic function. It can be a single
    //   element set, thought this is uncommon in practice, since concretization
    //   of a polymorphic function to a single concrete function is unnecessary.
    if (function_obj->poly_fn().concrete_functions().empty()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Function object: ", function_object_name,
          " is a set of concretized polymorphic function and should "
          "not be empty"));
    }
    for (const auto& concrete_function :
         function_obj->poly_fn().concrete_functions()) {
      functions.push_back(&concrete_function);
    }
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Function object: ", function_object_name,
                     " is not a polymorphic function or a function."));
  }

  for (const auto& function : functions) {
    if (!function->has_signature()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Function object: ", function_object_name, " has no signature."));
    }
  }
  return functions;
}

}  // namespace neptune
