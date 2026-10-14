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

#ifndef XLA_SERVING_RUNTIME_CONSTANTS_H_
#define XLA_SERVING_RUNTIME_CONSTANTS_H_

namespace xla_serving {

// Model version file name and known keys. The file is a text file named
// 'orbax_model_version.txt' with the following format:
//
// version: "0.0.1"
// mime_type: "application/protobuf; type=orbax_model_manifest.Manifest"
// manifest_file_path: "path/to/manifest.pb"
//
// The name is historical, but it is the one every reader in the ecosystem
// recognizes, so it is kept rather than tracking the project's renames.
inline constexpr char kModelVersionTxt[] = "orbax_model_version.txt";

inline constexpr char kModelManifestVersionKey[] = "version";
inline constexpr char kModelManifestMimeTypeKey[] = "mime_type";
inline constexpr char kModelManifestFilePathKey[] = "manifest_file_path";
inline constexpr char kModelManifestVersion[] = "0.0.1";
// The mime type of the manifest proto file. This names the on-disk format,
// which has not changed across the project's renames, so it deliberately does
// not track them. It is also the only spelling that appears in models already
// exported, and (unlike the proto package) it is not rewritten by Copybara, so
// it is identical internally and in OSS.
inline constexpr char kModelManifestMimeType[] =
    "application/protobuf; type=orbax_model_manifest.Manifest";

// Bundle orchestration proto file name.
inline constexpr char kBundleOrchestrationFileName[] =
    "bundle_orchestration.pb";
// The supplemental_info key for Orbax orchestration pipeline.
inline constexpr char kOrchestrationSupplementalName[] = "orchestration";
// Orbax checkpoint mime type expression.
inline constexpr char kOrbaxCheckpointMimeType[] =
    "application/x.orbax-checkpoint";
// Orbax orchestration mime type expression.
inline constexpr char kOexOrchestrationMultiPipelineMimeType[] =
    "application/protobuf; type=xla_saved_model_orchestration.Pipelines";
inline constexpr char kOexOrchestrationMultiPipelineMimeTypeLegacy[] =
    "application/protobuf; type=orbax_export_orchestration.Pipelines";
// StableHLO mime type expression.
inline constexpr char kStableHloMimeType[] = "application/x.mlir-stablehlo";
// XLA compile options mime type expression.
inline constexpr char kXlaCompileOptionsMimeType[] =
    "application/protobuf; type=xla_saved_model_manifest.CompileOptionsProtoMap";
inline constexpr char kXlaCompileOptionsMimeTypeLegacy[] =
    "application/protobuf; type=orbax_model_manifest.CompileOptionsProtoMap";
// TF SavedModel supplemental name.
inline constexpr char kTfSavedModelSupplementalName[] =
    "tensorflow_saved_model";
// TF SavedModel mime type expression.
inline constexpr char kTfSavedModelMimeType[] =
    "application/x.tensorflow-saved-model";
inline constexpr char kTfConcreteFunctionHandleMimeType[] =
    "application/protobuf; "
    "type=xla_saved_model_tf_concrete_function_handle.TfConcreteFunctionHandle";
inline constexpr char kTfConcreteFunctionHandleMimeTypeLegacy[] =
    "application/protobuf; "
    "type=orbax_model_tf_concrete_function_handle.TfConcreteFunctionHandle";

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_CONSTANTS_H_
