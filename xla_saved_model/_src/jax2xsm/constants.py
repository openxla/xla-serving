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

"""Constants used in jax2xsm."""

# XLA compilation options.
XLA_COMPILE_OPTIONS = 'xla_compile_options'

# TODO: Consolidate constants related to mime types in a common library, so that both obm and jsv can use them.
# XLA compile options mime type.
XLA_COMPILE_OPTIONS_MIME_TYPE = (
    'application/protobuf; type=neptune_model_manifest.CompileOptionsProtoMap'
)

# JAX specific info constants.
JAX_SPECIFIC_INFO = 'jax_specific_info'
CURRENT_JAX_SUPPLEMENTAL_MIME_TYPE = (
    'application/protobuf; type=neptune_model_jax_supplemental.Function'
)
CURRENT_JAX_SUPPLEMENTAL_VERSION = '0.0.1'
