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

"""Python ops and modules related to CpuEmbeddingLookupTransform."""

from data.kernels.cpu_embedding import cpu_embedding_module

MANIFEST_ASSET_NAME = cpu_embedding_module.MANIFEST_ASSET_NAME
serialize = cpu_embedding_module.serialize

CpuEmbeddingResource = cpu_embedding_module.CpuEmbeddingResource
CpuEmbeddingModule = cpu_embedding_module.CpuEmbeddingModule
