#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_MANIFEST_UTILS_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_MANIFEST_UTILS_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "protos/manifest.pb.h"  // from @neptune
#include "protos/orchestration.pb.h"  // from @neptune
#include "protos/type.pb.h"  // from @neptune
#include "runtime/core/flat_signature.h"  // from @neptune
#include "runtime/core/stablehlo_utils.h"  // from @neptune
#include "runtime/core/type.h"  // from @neptune
#include "xla/pjrt/proto/compile_options.pb.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla

namespace neptune {

// Device assignment information from the manifest.
struct AssignedDevices {
  // The device assignment by coords in the form of [x,y,z,...,core_on_chip].
  std::optional<std::vector<int>> device_assignment;
  int num_replicas = 1;
  int num_cores_per_replica = 1;
};

// Converts an `neptune_model_manifest::Shape` to `ArgShape`.
absl::StatusOr<ArgShape> ToArgShape(const neptune_model_type::Shape& shape);

// Flattens the types in a manifest into a vector of leaf types.
absl::Status FlattenType(
    const neptune_model_type::Type& type,
    std::vector<neptune_model_type::LeafType>& flatten_types);

// Gets the input spec of a function.
absl::StatusOr<std::vector<ArgSpec>> GetInputSpec(
    const neptune_model_manifest::Function& func);

// Gets the output spec of a function.
absl::StatusOr<std::vector<ArgSpec>> GetOutputSpec(
    const neptune_model_manifest::Function& func);

// Returns error if the mime type of `unstructured_data` is not
// `allowed_mime_type`.  A `description` of the `unstructured_data` should be
// provided to add more context for the error message, for example, "Checkpoint
// value". For `description`, see `CheckMimeType`.
absl::Status CheckMimeType(
    const neptune_model_manifest::UnstructuredData& unstructured_data,
    absl::Span<const absl::string_view> allowed_mime_types,
    absl::string_view description);

// Returns error if the data type of `unstructured_data` is not
// `allowed_data_type`. See `CheckMimeType` for `description`.
absl::Status CheckDataType(
    const neptune_model_manifest::UnstructuredData& unstructured_data,
    neptune_model_manifest::UnstructuredData::DataCase allowed_data_type,
    absl::string_view description);

// Returns error if there is a mismatch between the mime/data type of
// `neptune_model_manifest::UnstructuredData` and the expected mime/data type.
// For `description`, see `CheckMimeType`.
absl::Status CheckMimeAndDataType(
    const neptune_model_manifest::UnstructuredData& unstructured_data,
    absl::Span<const absl::string_view> allowed_mime_types,
    neptune_model_manifest::UnstructuredData::DataCase allowed_data_type,
    absl::string_view description);

// Gets the model's manifest given the `model_path`.
absl::StatusOr<neptune_model_manifest::Manifest> GetManifest(
    absl::string_view model_path);

// Gets the absolute file system path of the `file_system_location` given the
// `model_path`. The `file_system_location` should have a string path and the
// string path should be relative to the `model_path`. For now, we intentionally
// do not support absolute path in `file_system_location` to avoid potential
// issues.
absl::StatusOr<std::string> GetAbsoluteFileSystemPath(
    absl::string_view model_path,
    const neptune_model_manifest::FileSystemLocation& file_system_location);

// Flattens the types in a manifest into a vector of leaf types. It is
// necessary to flatten the types because the stable hlo fundamentally only
// support list of arguments.
absl::Status FlattenType(
    const neptune_model_type::Type& type,
    std::vector<neptune_model_type::LeafType>& flatten_types);

// Converts an `neptune_model_type::TensorType` to a `jsv::ArgSpec`.
absl::StatusOr<ArgSpec> ConvertToArgSpec(
    const neptune_model_type::TensorType& type);

// Gets the `FlatSignature` from function's input signature `type`. Each
// `MetaArgSpec` in the flat signature will not have the `name` field set. Input
// has args-kwargs pattern and may have offset. For example, in JAX, the offset
// in positional arguments is 1, which means that the first argument is always
// the tree representing params and we do not include it in the flat signature.
absl::StatusOr<FlatSignature> GetInputFlatSignature(
    const neptune_model_type::Type& type, int offset = 0);

// Gets the `FlatSignature` from function's output signature `type`. Each
// `MetaArgSpec` in the flat signature will not have the `name` field set.
absl::StatusOr<FlatSignature> GetOutputFlatSignature(
    const neptune_model_type::Type& type);

// Gets the `FlatSignature` from the `named_signature` in orchestration.
absl::StatusOr<FlatSignature> GetFlatSignatureFromNamedSignature(
    const neptune_model_orchestration::NamedTensorTypeList& inputs_or_outputs);

// Gets the assigned devices from the manifest if `compile_options` is not null,
// use `compile_options` for setting `num_cores_per_replica` and `num_replicas`.
absl::StatusOr<AssignedDevices> GetAssignedDevices(
    const neptune_model_manifest::Manifest& manifest,
    const neptune_model_manifest::Function& stable_hlo_func,
    const xla::CompileOptionsProto* absl_nullable compile_options,
    bool ignore_device_assignment_in_manifest = false);

// Gets the XLA compile options from the stablehlo function.
// When there are multiple stablehlo functions (i.e., multiple concretized
// stablehlo functions for a polymorphic function), the compile options must be
// consistent across them, otherwise an error will be returned.
absl::StatusOr<std::optional<xla::CompileOptionsProto>> GetCompileOptions(
    absl::Span<const neptune_model_manifest::Function* const> stablehlo_funcs,
    absl::string_view platform_name);

// Gets the input names for a given pipeline from the orchestration file. The
// returned names are ordered and correspond to the pipeline's input arguments.
//
// NOTE: This provides a way to get the input names without loading the whole
// model. If the model/pipeline is already loaded, prefer retrieving the input
// names from the loaded model directly.
absl::StatusOr<std::vector<std::string>> GetInputNames(
    absl::string_view orchestration_path, absl::string_view pipeline_name);

// Gets the output names for a given pipeline from the orchestration file. The
// returned names are ordered and correspond to the pipeline's output values.
//
// NOTE: This provides a way to get the output names without loading the whole
// model. If the model/pipeline is already loaded, prefer retrieving the output
// names from the loaded model directly.
absl::StatusOr<std::vector<std::string>> GetOutputNames(
    absl::string_view orchestration_path, absl::string_view pipeline_name);

absl::StatusOr<neptune_model_orchestration::Pipelines> GetOexOrchestration(
    absl::string_view model_path,
    const neptune_model_manifest::Manifest& manifest);

// Gets either the `Function` or the `PolymorphicFunction` from the manifest.
// The `Function` should be a `StableHLO` or a `TFFunction`. The
// `PolymorphicFunction` should be a set of `StableHLO`s with concrete input and
// output shapes that are concretized from a JAX polymorphic function.
// Gets the StableHLO function(s) from the manifest. It can be
// - either a single `Function` with concrete input and output shapes,
// - or a single `Function` with symbolic input and output shapes, or
// - or a set of `Function`s with specific input/output shapes that are
//   concretized from a JAX polymorphic function.
absl::StatusOr<std::vector<const neptune_model_manifest::Function*>>
GetStableHloFunctionsByObjectName(

    const absl::flat_hash_map<std::string,
                              const neptune_model_manifest::TopLevelObject*>&
        manifest_objects,
    absl::string_view function_object_name);

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_MANIFEST_UTILS_H_
