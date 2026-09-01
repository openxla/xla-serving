# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Constants for the Neptune model CLI."""

CLI_VERSION = '0.0.1'

MIME_TYPE_MLIR_STABLEHLO = 'application/x.mlir-stablehlo'
MIME_TYPE_ORBAX_CHECKPOINT = 'application/x.orbax-checkpoint'

MIME_TYPE_PIPELINES = (
    'application/protobuf; type=neptune_orchestration.Pipelines'
)

MIME_TYPE_ORBAX_PIPELINES = (
    'application/protobuf; type=orbax_export_orchestration.Pipelines'
)

MIME_TYPE_TF_CONCRETE_FUNCTION = (
    'application/protobuf;'
    ' type=neptune_model_tf_concrete_function_handle.TfConcreteFunctionHandle'
)

MIME_TYPE_ORBAX_TF_CONCRETE_FUNCTION = (
    'application/protobuf;'
    ' type=orbax_model_tf_concrete_function_handle.TfConcreteFunctionHandle'
)

MIME_TYPE_TF_SAVED_MODEL = 'application/x.tensorflow-saved-model'

MIME_TYPE_XLA_COMPILE_OPTIONS = (
    'application/protobuf; type=neptune_model_manifest.CompileOptionsProtoMap'
)

MIME_TYPE_ORBAX_XLA_COMPILE_OPTIONS = (
    'application/protobuf; type=orbax_model_manifest.CompileOptionsProtoMap'
)

MIME_TYPE_ORBAX_JAX_SUPPLEMENTAL = (
    'application/protobuf; type=neptune_model_jax_supplemental.Function'
)


WELL_KNOWN_MIME_TYPE_DESCRIPTIONS = {
    MIME_TYPE_MLIR_STABLEHLO: 'MLIR StableHLO',
    MIME_TYPE_ORBAX_CHECKPOINT: 'Orbax Checkpoint',
    MIME_TYPE_PIPELINES: 'Orchestration Pipelines',
    MIME_TYPE_TF_CONCRETE_FUNCTION: 'TensorFlow ConcreteFunction',
    MIME_TYPE_TF_SAVED_MODEL: 'TensorFlow SavedModel',
    MIME_TYPE_XLA_COMPILE_OPTIONS: 'XLA Compile Options',

    # Types for the legacy Orbax Model format.
    MIME_TYPE_ORBAX_PIPELINES: 'Orchestration Pipelines',
    MIME_TYPE_ORBAX_TF_CONCRETE_FUNCTION: 'TensorFlow ConcreteFunction',
    MIME_TYPE_ORBAX_XLA_COMPILE_OPTIONS: 'XLA Compile Options',
    MIME_TYPE_ORBAX_JAX_SUPPLEMENTAL: 'JAX Supplemental Function',
}

# Register flags for printing details for some types, typically
# ones that don't fit into the default details output.
MIME_TYPE_FLAGS = {
    MIME_TYPE_XLA_COMPILE_OPTIONS: 'xla_compile_options',
    MIME_TYPE_ORBAX_JAX_SUPPLEMENTAL: 'jax_specific_info',
    MIME_TYPE_ORBAX_XLA_COMPILE_OPTIONS: 'xla_compile_options',
}
