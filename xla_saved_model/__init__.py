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

"""Defines exported symbols for XLA SavedModel."""

# TODO: Add exports.

import importlib.metadata

from xla_saved_model._src.core import aot_recompiler
from xla_saved_model._src.core import offline_compiler
from xla_saved_model._src.export import jax_model
from xla_saved_model._src.export import main_lib
from xla_saved_model._src.export import serving_config
from xla_saved_model._src.export import tensorflow_model
from xla_saved_model._src.jax2xsm import typing

recompile = aot_recompiler.recompile
TargetConfig = offline_compiler.TargetConfig
JaxModel = jax_model.JaxModel
save = main_lib.save
ServingConfig = serving_config.ServingConfig
TensorflowModel = tensorflow_model.TensorflowModel
ApplyFnBundle = typing.ApplyFnBundle

__all__ = [
    "ApplyFnBundle",
    "JaxModel",
    "ServingConfig",
    "TargetConfig",
    "TensorflowModel",
    "recompile",
    "save",
    "serving_config",
]


try:
  __version__ = importlib.metadata.version("xla-saved-model")
except importlib.metadata.PackageNotFoundError:
  __version__ = "unknown"
