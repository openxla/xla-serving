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

"""Stitching tools for XLA SavedModel."""

from xla_saved_model.tools.stitching.apply_function_loader import get_params_structure
from xla_saved_model.tools.stitching.apply_function_loader import load_apply_fn
from xla_saved_model.tools.stitching.extra_asset_loader import find_extra_asset_path
from xla_saved_model.tools.stitching.extra_asset_loader import load_extra_asset
from xla_saved_model.tools.stitching.weights_loader import find_checkpoint_path
from xla_saved_model.tools.stitching.weights_loader import load_params

# `tf_function_loader` is not imported here because it needs TensorFlow, which
# is an optional dependency.

__all__ = [
    "find_checkpoint_path",
    "find_extra_asset_path",
    "get_params_structure",
    "load_apply_fn",
    "load_extra_asset",
    "load_params",
]
