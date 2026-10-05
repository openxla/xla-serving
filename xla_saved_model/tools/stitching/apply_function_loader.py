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

"""Tools for loading XLA SavedModel functions back to JAX."""

import pathlib

import jax
import jaxtyping

from xla_saved_model.protos import jax_supplemental_pb2
from xla_saved_model._src import xsm2jax
from xla_saved_model._src.core import manifest_utils
from xla_saved_model._src.core import persistence_lib
from xla_saved_model._src.jax2xsm import constants as jax2xsm_constants


def load_apply_fn(
    model_dir: pathlib.Path | str,
    function_name: str,
) -> xsm2jax.ExportedApplyFn:
  """Loads an XLA SavedModel function from a model directory as a JAX ApplyFn.

  Args:
    model_dir: Directory where the XLA SavedModel was saved. Can be a
      `pathlib.Path` or string.
    function_name: Name of the function in the model manifest to load.

  Returns:
    A JAX `ApplyFn` (wrapped in `ExportedApplyFn`) that can be called directly
    as `fn(params, inputs)`.

  Raises:
    FileNotFoundError: If the model directory or manifest cannot be found.
    KeyError: If the specified function name does not exist in the manifest.
    ValueError: If the object is not a function, lacks a StableHLO body, or
      lacks JAX supplemental information with standard MIME type.
  """
  model_dir = pathlib.Path(model_dir)

  manifest = persistence_lib.load(str(model_dir))

  if function_name not in manifest.objects:
    available_fns = [
        k for k, v in manifest.objects.items() if v.HasField("function")
    ]
    raise KeyError(
        f"Function '{function_name}' not found in manifest. "
        f"Available functions: {available_fns}"
    )

  obj = manifest.objects[function_name]
  match obj.WhichOneof("case"):
    case "function":
      manifest_function = obj.function
    case other_case:
      raise ValueError(
          f"Object '{function_name}' is not a function (got '{other_case}')."
      )

  if not manifest_function.body.HasField("stable_hlo_body"):
    raise ValueError(
        f"Function '{function_name}' does not have a StableHLO body."
    )
  stable_hlo_body = manifest_function.body.stable_hlo_body

  jax_supplemental_data = None
  for data in stable_hlo_body.supplemental_info.values():
    if data.mime_type == jax2xsm_constants.CURRENT_JAX_SUPPLEMENTAL_MIME_TYPE:
      jax_supplemental_data = data
      break

  if jax_supplemental_data is None:
    raise ValueError(
        f"Function '{function_name}' does not contain JAX supplemental info"
        " with standard MIME type"
        f" '{jax2xsm_constants.CURRENT_JAX_SUPPLEMENTAL_MIME_TYPE}'."
    )

  raw_bytes = manifest_utils.read_unstructured_data_bytes(
      jax_supplemental_data, str(model_dir), manifest
  )
  jax_supplemental_proto = jax_supplemental_pb2.Function.FromString(raw_bytes)

  return xsm2jax.xsm_fn_to_apply_fn(manifest_function, jax_supplemental_proto)


def get_params_structure(
    apply_fn: xsm2jax.ExportedApplyFn,
) -> jaxtyping.PyTree:
  """Returns the structure of a loaded function's `params` argument.

  A loaded function is called as `apply_fn(params, inputs)`; this returns the
  pytree structure of its first positional argument, `params`, with each leaf
  as a `jax.ShapeDtypeStruct`. `inputs` is not included.

  The structure comes from the function's recorded input signature, so custom
  nodes such as `qwix.QArray` are rebuilt only if the model recorded them and
  the module defining each one is imported before `load_apply_fn` is called.
  Otherwise they appear as plain lists.

  Args:
    apply_fn: The function as returned by `load_apply_fn`.

  Returns:
    The `params` pytree, with each leaf a `jax.ShapeDtypeStruct`. Shapes may
    contain symbolic dimensions.
  """
  exported = apply_fn.exported
  structs = [
      jax.ShapeDtypeStruct(aval.shape, aval.dtype) for aval in exported.in_avals
  ]
  (params, _), _ = jax.tree.unflatten(exported.in_tree, structs)
  return params


__all__ = [
    "get_params_structure",
    "load_apply_fn",
]
