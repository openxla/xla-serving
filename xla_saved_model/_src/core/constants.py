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
    'application/protobuf; type=neptune_orchestration.Pipelines'
)
ORCHESTRATION_MULTI_PIPELINE_VERSION = '0.0.1'
# The filename of the orchestration proto file relative to the save
# directory.
DEFAULT_SUPPLEMENTAL_FILENAME = 'orchestration.pb'
JAX_SPECIFIC_INFO = 'jax_specific_info'
