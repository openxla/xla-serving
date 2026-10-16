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

"""Constants for the XLA SavedModel."""

################################################################################
# SHLO constants
################################################################################

SHLO_MIME_TYPE = 'application/x.mlir-stablehlo'
SHLO_VERSION = '1.0'

################################################################################
# TF SavedModel constants
################################################################################

# The directory, relative to the model directory, of the TF SavedModel that the
# model's TF functions are saved in, and served from.
TF_SAVED_MODEL_SUB_DIR = 'tf_saved_model/'

# The directory, relative to the model directory, of the TF SavedModel as saved
# before it was converted into `TF_SAVED_MODEL_SUB_DIR`. Only models saved with
# a conversion step, and asked to keep the original, have it. Unlike the
# converted TF SavedModel, it keeps the object graph, so loading it restores
# every attribute of the saved module, not only its signatures. A model saved
# without a conversion step has no such copy: `TF_SAVED_MODEL_SUB_DIR` is itself
# unconverted.
UNCONVERTED_TF_SAVED_MODEL_SUB_DIR = 'unconverted_tf_saved_model/'

################################################################################
# Executable constants
################################################################################

# The mime type of the compiled IFRT / PJRT executable payload in the manifest.
IFRT_EXECUTABLE_MIME_TYPE = 'application/x-ifrt-executable'

################################################################################
# Manifest model format constants
################################################################################

# The filename of the model version metadata file relative to the save
# directory. The name is historical, but it is the one every reader in the
# ecosystem recognizes, so it is kept rather than tracking the project's
# renames.
MODEL_VERSION_FILENAME = 'orbax_model_version.txt'

# The file path of the manifest proto file relative to the save directory.
MANIFEST_FILE_PATH = 'manifest.pb'

# The file path of the modifications history file relative to the save
# directory.
HISTORY_FILENAME = 'history.txtpb'

# The version of the manifest.
MANIFEST_VERSION = '0.0.1'

# The mime type of the manifest proto file. This names the on-disk format,
# which has not changed across the project's renames, so it deliberately does
# not track them. It is also the only spelling that appears in models already
# exported, and (unlike the proto package) it is not rewritten by Copybara, so
# it is identical internally and in OSS.
MANIFEST_MIME_TYPE = (
    'application/protobuf; type=orbax_model_manifest.Manifest'
)

################################################################################
# Orchestration constants
################################################################################

ORCHESTRATION_MULTI_PIPELINE_MIME_TYPE = (
    'application/protobuf; type=xla_saved_model_orchestration.Pipelines'
)
ORCHESTRATION_MULTI_PIPELINE_VERSION = '0.0.1'
# The filename of the orchestration proto file relative to the save
# directory.
DEFAULT_SUPPLEMENTAL_FILENAME = 'orchestration.pb'
JAX_SPECIFIC_INFO = 'jax_specific_info'
