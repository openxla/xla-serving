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

"""A JAX model, consisting of an apply_fn and its parameters."""

from collections.abc import Mapping, Sequence
import pathlib

import jax
import jaxtyping
import orbax.checkpoint as ocp

from xla_saved_model._src import jax2xsm
from xla_saved_model._src.core import value
from xla_saved_model._src.export import constants
from xla_saved_model._src.export.constants import DEFAULT_METHOD_KEY
from xla_saved_model._src.jax2xsm.typing import ApplyFnBundle

DEFAULT_MODEL_NAME: str = 'jax_model'


class JaxModel:
  """A JAX model, consisting of an apply_fn and its parameters.

  Attributes:
    name: The name of the model.
    apply_fn_bundle: The function bundle for applying the model, or a mapping of
      such bundles.
    params: The parameters of the model.
    checkpoint_path: The path to an existing checkpoint of the model parameters.
    jax_mesh: Mesh for the model.
  """

  def __init__(
      self,
      apply_fn_bundle: ApplyFnBundle | Mapping[str, ApplyFnBundle],
      params: jaxtyping.PyTree | None = None,
      checkpoint_path: str | None = None,
      jax_mesh: jax.sharding.Mesh | None = None,
      name: str = DEFAULT_MODEL_NAME,
  ):
    if isinstance(apply_fn_bundle, Mapping):
      self._apply_fn_bundle = apply_fn_bundle
    else:
      self._apply_fn_bundle = {DEFAULT_METHOD_KEY: apply_fn_bundle}
    self._params = params
    self._checkpoint_path = checkpoint_path
    self._jax_mesh = jax_mesh
    if not name:
      raise ValueError('JaxModel name must not be empty')
    self._name = name

  @property
  def name(self) -> str:
    return self._name

  @property
  def weights_name(self) -> str:
    """The default weights name for the model checkpoint."""
    return f'{self._name}_{constants.WEIGHTS_NAME}'

  @property
  def has_weights(self) -> bool:
    """Whether this model has parameters or an external checkpoint path."""
    return self._params is not None or self._checkpoint_path is not None

  @property
  def params(self) -> jaxtyping.PyTree | None:
    return self._params

  @property
  def apply_fn_bundles(self) -> Mapping[str, ApplyFnBundle]:
    """A mapping from method keys to ApplyFnBundles."""
    return self._apply_fn_bundle

  def method_keys(self) -> Sequence[str]:
    return list(self._apply_fn_bundle.keys())

  @property
  def jax_mesh(self) -> jax.sharding.Mesh | None:
    return self._jax_mesh

  def save_checkpoint(
      self,
      export_path: str,
      checkpoint_path: str,
      weights_name: str,
  ) -> Mapping[str, value.ExternalValue]:
    """Saves the JAX model parameters to an Orbax checkpoint or references an existing checkpoint.

    If `checkpoint_path` was provided when initializing `JaxModel`, the external
    checkpoint path is registered in the manifest without saving parameters.
    Otherwise, if `params` are present, they are saved to `checkpoint_path` within
    `export_path`.

    Args:
      export_path: The root directory where the XLA SavedModel is being exported.
      checkpoint_path: The relative subdirectory path within `export_path` where
        the Orbax checkpoint should be saved if `checkpoint_path` was not provided
        to the model.
      weights_name: The manifest object key under which this checkpoint's
        external value will be registered.

    Returns:
      A mapping from `weights_name` to the corresponding ExternalValue
      object representing the saved checkpoint, or an empty dict if the model
      has no parameters and no checkpoint path.
    """
    if self._checkpoint_path is not None:
      return {
          weights_name: jax2xsm.convert_path_to_value(
              self._checkpoint_path,
              constants.ORBAX_CHECKPOINT_MIME_TYPE,
          )
      }
    if self._params is None:
      return {}
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    checkpointer.save(
        pathlib.Path(export_path) / checkpoint_path,
        self._params,
        force=True,
    )
    return {
        weights_name: jax2xsm.convert_path_to_value(
            checkpoint_path,
            constants.ORBAX_CHECKPOINT_MIME_TYPE,
        )
    }
