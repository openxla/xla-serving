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

"""Utility functions for OBM test cases."""

import collections
from collections.abc import Callable, Iterable, Mapping
import copy
import operator
import pathlib
from typing import Any, cast
import unittest

from absl.testing import parameterized
from google.protobuf import descriptor
from google.protobuf import message
from google.protobuf import text_format
import jax
from jax import export as jax_export
from jax.experimental import topologies
import jax.numpy as jnp
import jaxtyping

from xla_saved_model.protos import manifest_pb2
from xla_saved_model._src.core import file_utils
from xla_saved_model._src.export import jax_model
from xla_saved_model._src.export import main_lib
from xla_saved_model._src.export import serving_config


# TODO: Remove skip once jaxlib >= 0.11.2 is deployed.
def get_topology_desc(
    topology_name: str, platform: str
) -> topologies.TopologyDescription:
  """Safely retrieves topology descriptor or skips if plugin is missing."""
  try:
    return topologies.get_topology_desc(topology_name, platform=platform)
  except (NotImplementedError, RuntimeError) as e:
    raise unittest.SkipTest(
        f"{platform.upper()} topology support not installed in this binary: {e}"
    ) from e


# TODO: Remove skip once jaxlib >= 0.11.2 is deployed.
def get_serialized_topology(topology_name: str, platform: str) -> bytes:
  """Retrieves serialized topology bytes or skips if unsupported."""
  topo = get_topology_desc(topology_name, platform=platform)
  try:
    return topo.serialize()
  except (AttributeError, RuntimeError) as e:
    raise unittest.SkipTest(
        f"{platform.upper()} topology serialization not supported in this"
        f" binary: {e}"
    ) from e


def require_attached_hardware(platform: str) -> None:
  """Verifies physical accelerator hardware is present or raises SkipTest."""
  try:
    jax.devices(platform)
  except (RuntimeError, ValueError) as e:
    raise unittest.SkipTest(
        f"Requires attached {platform.upper()} hardware: {e}"
    ) from e


def create_sample_autotune_file(
    parent_dir: str, filename: str = "autotune.textproto"
) -> str:
  """Creates a sample autotune database file and returns its path."""
  path = str(pathlib.Path(parent_dir) / filename)
  with file_utils.open_file(path, "w") as f:
    f.write("# autotune database placeholder\n")
  return path


def create_sample_autotune_dir(
    parent_dir: str, dirname: str = "autotune_dir"
) -> str:
  """Creates a sample autotune database directory and returns its path."""
  path = str(pathlib.Path(parent_dir) / dirname)
  file_utils.mkdir_p(path)
  return path


def create_serialized_topology_file(
    topology_name: str,
    parent_dir: str,
    platform: str,
    filename: str = "topology.pb",
) -> str:
  """Serializes a topology and writes it to a file, returning the file path."""
  serialized_bytes = get_serialized_topology(topology_name, platform=platform)
  topo_file = str(pathlib.Path(parent_dir) / filename)
  with file_utils.open_file(topo_file, "wb") as f:
    f.write(serialized_bytes)
  return topo_file


def get_vhlo_portable_artifact() -> bytes:
  """Generates valid VHLO portable artifact using jax.export."""
  exported = jax_export.export(jax.jit(lambda x: x + 1.0))(
      jax.ShapeDtypeStruct((2, 2), jnp.float32)
  )
  return exported.mlir_module_serialized


def get_raw_text_mlir() -> bytes:
  """Returns raw text MLIR module bytes."""
  return (
      b"module { func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {"
      b" return %arg0 : tensor<2x2xf32> } }"
  )


def get_stablehlo_mlir() -> bytes:
  """Generates valid StableHLO MLIR bytecode dynamically using JAX."""
  lowered = jax.jit(lambda x: x + 1.0).lower(jnp.ones((2, 2), jnp.float32))
  module: Any = lowered.compiler_ir(dialect="stablehlo")
  if module is None:
    raise RuntimeError("Failed to lower JAX function to StableHLO dialect.")
  return module.operation.get_asm(binary=True)


def create_test_xla_saved_model(
    export_path: str,
    *,
    save_shlo_to_file: bool = False,
) -> str:
  """Exports an XLA SavedModel test model with genuine JAX math (sin(x) + w)."""

  def forward(params, inputs):
    return {"out": jnp.sin(inputs["x"]) + params["w"]}

  params = {"w": jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)}
  spec = jax.ShapeDtypeStruct((2, 2), jnp.float32)

  bundle = jax_model.ApplyFnBundle(apply_fn=forward)
  model = jax_model.JaxModel(
      apply_fn_bundle=bundle, params=params, name="model"
  )
  cfg = serving_config.ServingConfig(
      serving_signature="serving_default",
      input_signature={"x": spec},
  )

  main_lib.save(
      models=model,
      export_path=export_path,
      serving_configs=[cfg],
      save_shlo_to_file=save_shlo_to_file,
  )
  return export_path


def _shape_dtype_equal(tree1: Any, tree2: Any) -> bool:
  return tuple(tree1.shape) == tuple(tree2.shape) and tree1.dtype == tree2.dtype


def _clear_fields(msg: message.Message, fields: Iterable[str]) -> None:
  """Recursively clears fields matching dot-separated paths in a proto."""
  if not hasattr(msg, "ClearField"):
    return

  clear_map = collections.defaultdict(list)
  for field in fields:
    if not field:
      continue
    prefix, _, suffix = field.partition(".")
    if suffix:
      clear_map[prefix].append(suffix)
    else:
      msg.ClearField(prefix)

  for prefix, suffixes in clear_map.items():
    fd = msg.DESCRIPTOR.fields_by_name.get(prefix)
    if fd is None:
      continue

    # We only recurse if the field is a message/group descriptor
    if fd.type == descriptor.FieldDescriptor.TYPE_MESSAGE:
      attr_val = getattr(msg, prefix)
      if isinstance(attr_val, Mapping):
        # Handle map fields
        for k in attr_val:
          _clear_fields(attr_val[k], suffixes)
      elif fd.is_repeated:
        # Handle repeated fields
        for entry in attr_val:
          _clear_fields(entry, suffixes)
      else:
        # Handle singular message fields
        if msg.HasField(prefix):
          _clear_fields(attr_val, suffixes)


class XsmTestCase(parameterized.TestCase):  # pylint: disable=g-doc-args
  """Base class for OBM test cases.

  Provides utility methods for testing OBM code.
  """

  def assertTreesEqual(
      self,
      tree1: jaxtyping.PyTree,
      tree2: jaxtyping.PyTree,
      leaf_eq: Callable[[Any, Any], bool] = operator.eq,
  ):  # pylint: disable=invalid-name
    """Asserts that two pytrees are equal."""
    if jax.tree.structure(tree1) != jax.tree.structure(tree2):
      raise AssertionError(
          f"Trees have different structures\ntree1:\n{tree1},\ntree2:\n{tree2}"
      )

    eq_tree = jax.tree.map(leaf_eq, tree1, tree2)
    if not all(jax.tree.leaves(eq_tree)):
      raise AssertionError(
          f"Trees are not equal\ntree1:\n{tree1},\ntree2:\n{tree2}"
      )

  def assertShapeDtypeEqual(
      self,
      tree1: jaxtyping.PyTree,
      tree2: jaxtyping.PyTree,
  ):  # pylint: disable=invalid-name
    """Asserts that two pytrees have leaves with the same shape and dtype."""
    self.assertTreesEqual(tree1, tree2, leaf_eq=_shape_dtype_equal)

  def assertProtosEqual(
      self,
      expected: str | message.Message,
      actual: message.Message,
      *,
      ignored_fields: Iterable[str] | None = None,
  ):  # pylint: disable=invalid-name
    """Asserts that two protos are equal."""

    if isinstance(expected, str):
      expected_msg = actual.__class__()
      text_format.Parse(expected, expected_msg)
      parsed_expected = cast(message.Message, expected_msg)
    else:
      parsed_expected = copy.deepcopy(cast(message.Message, expected))

    parsed_actual = copy.deepcopy(actual)

    if ignored_fields:
      _clear_fields(parsed_expected, ignored_fields)
      _clear_fields(parsed_actual, ignored_fields)

    self.assertEqual(
        text_format.MessageToString(parsed_expected),
        text_format.MessageToString(parsed_actual),
    )

  def assert_compiled_executable_valid(
      self,
      model_dir: str,
      executable: manifest_pb2.CompiledExecutableBody,
  ):
    """Asserts that the compiled executable binary file exists and is non-empty."""
    bin_path = str(
        pathlib.Path(model_dir)
        / executable.aot_executable.file_system_location.string_path
    )
    self.assertTrue(
        file_utils.exists(bin_path), f"Binary does not exist at: {bin_path}"
    )
    self.assertGreater(
        pathlib.Path(bin_path).stat().st_size,
        0,
        f"Binary at {bin_path} is empty",
    )
