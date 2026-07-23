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

import jax
import jaxtyping

from neptune_model._src.jax2npt.typing import ApplyFnBundle


SERVING_DEFAULT_KEY = 'serving_default'


class JaxModel:
  """A JAX model, consisting of an apply_fn and its parameters.

  Exactly one of `params` or `checkpoint_path` must be provided.

  Attributes:
    name: The name of the model.
    apply_fn_bundle: The function bundle for applying the model, or a mapping of
      such bundles.
    params: The parameters of the model.
    jax_mesh: Mesh for the model.
  """

  def __init__(
      self,
      apply_fn_bundle: ApplyFnBundle | Mapping[str, ApplyFnBundle],
      params: jaxtyping.PyTree | None = None,
      jax_mesh: jax.sharding.Mesh | None = None,
      name: str = 'jax_model',
  ):
    if isinstance(apply_fn_bundle, Mapping):
      self._apply_fn_bundle = apply_fn_bundle
    else:
      self._apply_fn_bundle = {SERVING_DEFAULT_KEY: apply_fn_bundle}
    self._params = params
    self._jax_mesh = jax_mesh
    self._name = name

  @property
  def name(self) -> str:
    return self._name

  @property
  def params(self) -> jaxtyping.PyTree | None:
    return self._params

  def method_keys(self) -> Sequence[str]:
    return list(self._apply_fn_bundle.keys())

  @property
  def jax_mesh(self) -> jax.sharding.Mesh | None:
    return self._jax_mesh

  @property
  def apply_fn_bundles(self) -> Mapping[str, ApplyFnBundle]:
    return self._apply_fn_bundle
