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

r"""Utility to export Orbax model."""

import abc
from collections.abc import Sequence
import os
from etils import epath
import jax
import jaxtyping
import orbax.checkpoint as ocp


class ModelExporter(abc.ABC):
  """Exports a JAX model to Orbax model format."""

  def __init__(self, output_path: epath.Path, extra_files: Sequence[str] = ()):
    """Initializes the exporter.

    Args:
      output_path: The path to exported model.
      extra_files: The extra files to be copied to the output directory.
    """
    self._output_path = output_path
    self._create_output_dir()
    self._ckptr_subdir = 'ckpt'
    self._extra_files = [epath.Path(f) for f in extra_files]

  def _create_output_dir(self):
    """Creates the output directory."""
    if self._output_path.exists():
      self._output_path.rmtree()
    epath.Path(self._output_path).mkdir(parents=True, exist_ok=False)

  def _copy_extra_files(self):
    """Copies extra files to the output directory."""
    for extra_file in self._extra_files:
      extra_file.copy(self._output_path.joinpath(extra_file.name))

  def _save_params(self, params: jaxtyping.PyTree):
    """Writes params to checkpoint under `ckpt` subdir."""
    ckptr = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    ckptr_dir = os.path.abspath(
        os.path.join(self._output_path, self._ckptr_subdir)
    )
    ckptr.save(
        directory=ckptr_dir,
        args=ocp.args.StandardSave(
            item=params,
            save_args=jax.tree_util.tree_map(lambda p: ocp.SaveArgs(), params),
        ),
    )
    # Run restore to verify the ckpt can be loaded.
    _ = ckptr.restore(ckptr_dir)

  @abc.abstractmethod
  def _create_mesh(self) -> jax.sharding.Mesh:
    """Creates the mesh for sharding."""

  @abc.abstractmethod
  def export(self) -> None:
    """Exports a model."""
