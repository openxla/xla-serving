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

"""Converts TF functions to Neptune functions (allowing TF resources)."""

from collections.abc import Callable, Mapping, MutableMapping, Sequence
import copy
import dataclasses
import functools
import inspect
import os
from typing import Any, NamedTuple, cast

import jax
import jaxtyping
import tensorflow as tf

from neptune_model._src.core import file_utils
from neptune_model._src.core import persistence_lib
from neptune_model._src.core import serializable_function
from neptune_model._src.core import shlo_type
from neptune_model._src.core import unstructured_data
from neptune_model._src.tf2npt import type_conversion
from neptune_model.protos import manifest_pb2
from neptune_model.protos import tf_concrete_function_handle_pb2
# pylint: disable=g-direct-tensorflow-import
from tensorflow.core.protobuf import saved_model_pb2
# pylint: enable=g-direct-tensorflow-import

TF_CONCRETE_FUNCTION_HANDLE_MIME_TYPE = (
    'application/protobuf;'
    ' type=neptune_model_tf_concrete_function_handle.TfConcreteFunctionHandle'
)
TF_CONCRETE_FUNCTION_HANDLE_VERSION = '0.0.1'
NPT_TF_SAVED_MODEL_SUB_DIR = 'tf_saved_model/'
TF_SAVED_MODEL_SUPPLEMENTAL_NAME = 'tensorflow_saved_model'
SAVED_MODEL_MIME_TYPE = 'application/x.tensorflow-saved-model'
SAVED_MODEL_VERSION = '1.0'

_OUTPUT_NAME_PREFIX = 'output'
_PB_EXT_NAME = 'pb'


@dataclasses.dataclass(frozen=True)
class PreparedFunction:
  """TF function prepared for Neptune export.

  Attributes:
    concrete_fn: The concrete TF function (to be saved in the TF SavedModel).
    serializable_fn: The Neptune function generated from `concrete_fn`.
    defaults: The default values for `concrete_fn`. TF concrete functions do not
      keep track of the default values of the TF polymorhic functions they are
      concretized from, hence we need to store them separately.
  """

  concrete_fn: tf.types.experimental.ConcreteFunction
  serializable_fn: serializable_function.SerializableFunction
  defaults: dict[str, Any] | None = None


def _is_args_kwargs(tree: jaxtyping.PyTree) -> bool:
  """Checks if a TF signature tree is in the '(args, kwargs)' pattern."""
  return (
      isinstance(tree, Sequence)
      and len(tree) == 2
      and isinstance(tree[0], Sequence)
      and isinstance(tree[1], dict)
  )


def _to_args_kwargs(
    tf_signature: jaxtyping.PyTree[tf.TensorSpec],
) -> tuple[Sequence[tf.TensorSpec], dict[str, tf.TensorSpec]]:
  """Converts TF signature to the '(args, kwargs)' pattern."""
  if _is_args_kwargs(tf_signature):
    return tf_signature[0], tf_signature[1]

  if isinstance(tf_signature, Sequence):
    return tf_signature, {}

  return (tf_signature,), {}


def _to_npt_signature(
    tf_signature: jaxtyping.PyTree[tf.TensorSpec] | None,
) -> jaxtyping.PyTree[shlo_type.ShloTensorSpec] | None:
  return jax.tree.map(type_conversion.to_npt_spec, tf_signature)


def _get_defaults(
    fn: tf.types.experimental.PolymorphicFunction,
) -> dict[str, Any] | None:
  """Returns the default values for the given function."""
  source_fn = fn.python_function if hasattr(fn, 'python_function') else fn
  original_sig = inspect.signature(source_fn)

  defaults = {
      k: v.default
      for k, v in original_sig.parameters.items()
      if v.default is not inspect.Parameter.empty
  }
  return defaults


def _is_x64(dtype: tf.DType) -> bool:
  return dtype in (tf.float64, tf.complex128, tf.int64, tf.uint64)


def _has_x64_outputs(fn: tf.types.experimental.ConcreteFunction) -> bool:
  """Checks if the function has 64-bit outputs."""
  return any(
      isinstance(x, (tf.Tensor, tf.TensorSpec)) and _is_x64(x.dtype)
      for x in tf.nest.flatten(fn.structured_outputs)
  )


_X64_TO_X32 = {
    tf.float64: tf.float32,
    tf.complex128: tf.complex64,
    tf.int64: tf.int32,
    tf.uint64: tf.uint32,
}


def _suppress_x64(x: Any) -> Any:
  if not isinstance(x, tf.Tensor):
    return x
  new_dtype = _X64_TO_X32.get(x.dtype, None)
  if new_dtype is None:
    return x
  return tf.cast(x, new_dtype)


def _suppress_x64_outputs(fn: Callable[..., Any]) -> Callable[..., Any]:
  """Wraps the function to suppress 64-bit outputs."""

  @functools.wraps(fn)
  def wrapped(*args, **kwargs):
    outputs = fn(*args, **kwargs)
    return tf.nest.map_structure(_suppress_x64, outputs)

  return wrapped


def _get_concrete_function(
    fn: Callable[..., Any],
    input_signature: jaxtyping.PyTree | None = None,
) -> tuple[tf.types.experimental.ConcreteFunction, dict[str, Any] | None]:
  """Returns the concrete function and defaults for the given function."""

  if isinstance(fn, tf.types.experimental.ConcreteFunction):
    return fn, None

  if isinstance(fn, tf.types.experimental.PolymorphicFunction):
    tf_callable = cast(tf.types.experimental.PolymorphicFunction, fn)
    if tf_callable.input_signature is not None:
      input_signature = tf_callable.input_signature
  else:
    tf_callable = tf.function(fn)

  if input_signature is None:
    raise ValueError('Input signature is missing')

  defaults = _get_defaults(tf_callable)
  tf_input_signature = jax.tree.map(type_conversion.to_tf_spec, input_signature)
  args, kwargs = _to_args_kwargs(tf_input_signature)

  if (
      len(tf_callable.function_spec.fullargspec.args) == 1
      and len(args) > 1
      and not kwargs
  ):
    # If the function expects a single positional argument
    # we will pass the args as a single argument.
    return tf_callable.get_concrete_function(args), defaults

  return tf_callable.get_concrete_function(*args, **kwargs), defaults


def prepare_function(
    fn_name: str,
    fn: Callable[..., Any],
    *,
    input_signature: jaxtyping.PyTree | None = None,
    suppress_x64_outputs: bool = False,
) -> PreparedFunction:
  """Converts the TF concrete function to a Neptune function.

  The resulting Neptune function is effectively a reference to a function
  stored in TF SavedModel by `save_tf_functions`.

  Args:
    fn_name: The name to be used in the Neptune manifest to refer to the TF
      function.
    fn: The function to prepare.
    input_signature: The input signature to use for the TF function. If None,
      the input signature will be inferred from the function.
    suppress_x64_outputs: If True, casts 64-bit outputs to 32-bit.

  Returns:
    The Neptune function referring to the original TF function in the TF
    SavedModel.
  """
  initial_concrete_fn, defaults = _get_concrete_function(fn, input_signature)

  if suppress_x64_outputs and _has_x64_outputs(initial_concrete_fn):
    concrete_fn, _ = _get_concrete_function(
        _suppress_x64_outputs(fn),
        initial_concrete_fn.structured_input_signature,
    )
  else:
    concrete_fn = initial_concrete_fn

  input_names, _, _ = _flat_input_signature(concrete_fn)
  output_names = _output_names(concrete_fn)

  final_input_signature = concrete_fn.structured_input_signature
  final_output_signature = get_output_signature(concrete_fn)

  fn_handle = manifest_pb2.UnstructuredData(
      inlined_bytes=tf_concrete_function_handle_pb2.TfConcreteFunctionHandle(
          fn_name=fn_name,
          input_names=list(input_names),
          output_names=list(output_names),
      ).SerializeToString(),
      mime_type=TF_CONCRETE_FUNCTION_HANDLE_MIME_TYPE,
      version=TF_CONCRETE_FUNCTION_HANDLE_VERSION,
  )

  return PreparedFunction(
      serializable_fn=serializable_function.SerializableFunction(
          body=unstructured_data.UnstructuredDataWithExtName(
              proto=fn_handle,
              ext_name=_PB_EXT_NAME,
          ),
          input_signature=_to_npt_signature(final_input_signature),
          output_signature=_to_npt_signature(final_output_signature),
      ),
      concrete_fn=concrete_fn,
      defaults=defaults,
  )


def tf_saved_model_as_npt_supplemental(
    subdir: str,
) -> manifest_pb2.UnstructuredData:
  """An Neptune Model global supplemental pointing to a TF SavedModel (path).

  Args:
    subdir: the path to the TF SavedModel dir. This can be a path relative to
      the Neptune Model dir, such as 'tf_saved_model/', or an absolute path.

  Returns:
    An `manifest_pb2.UnstructuredData` which can be used as an Neptune Model
    global supplemental.
  """
  return manifest_pb2.UnstructuredData(
      file_system_location=manifest_pb2.FileSystemLocation(string_path=subdir),
      mime_type=SAVED_MODEL_MIME_TYPE,
      version=SAVED_MODEL_VERSION,
  )


def _make_dict_only_signature(
    names: Sequence[str], leaves: Sequence[tf.TensorSpec]
) -> Mapping[str, tf.TensorSpec]:
  """Creates a dict-only signature from names and leaves.

  Args:
    names: A sequence of names for the leaves.
    leaves: A sequence of tensor specs.

  Returns:
    A dict-only signature where keys are names and values are tensor specs.
  """
  return {
      k: tf.TensorSpec(
          shape=v.shape,
          dtype=v.dtype,
          name=None,
      )
      for k, v in zip(names, leaves)
  }


def _tree_to_dict(
    tree: jaxtyping.PyTree[tf.TensorSpec] | None,
    names: Sequence[str] | None,
) -> Mapping[str, tf.TensorSpec] | None:
  """Converts a TF signature tree to a dictionary if names are provided.

  If `names` is None or `tree` is None, returns None. If `names` is provided,
  `tree` is flattened and a dictionary is created by mapping names to leaves.

  Args:
    tree: The TF signature tree.
    names: An optional sequence of names for leaves.

  Returns:
    A dictionary mapping names to leaves, or None.
  """
  if not names or tree is None:
    return None
  flat = jax.tree.leaves(tree)
  return dict(zip(names, flat))


def _dict_to_tree(
    d: Mapping[str, tf.TensorSpec] | None,
    names: Sequence[str] | None,
    tree_def: jax.tree_util.PyTreeDef,
) -> jaxtyping.PyTree[tf.TensorSpec] | None:
  """Converts a dictionary to a TF signature tree if names are provided.

  If `names` is None, `d` is returned unchanged. If `names` is provided,
  leaves are extracted from `d` in the order of `names` and unflattened into a
  tree using `tree_def`.

  Args:
    d: A dictionary mapping names to leaves.
    names: An optional sequence of names for leaves.
    tree_def: The PyTreeDef for unflattening.

  Returns:
    A TF signature tree.
  """
  if not names or not d:
    return None
  flat = tuple(d[name] for name in names)
  return jax.tree.unflatten(tree_def, flat)


class SignatureFlat(NamedTuple):
  """The flattened form of a signature.

  Attributes:
    names: A sequence of names for the leaves.
    leaves: A sequence of tensor specs.
    tree_def: The PyTreeDef for unflattening.
  """

  names: Sequence[str]
  leaves: Sequence[tf.TensorSpec]
  tree_def: jax.tree_util.PyTreeDef


def _flat_input_signature(
    fn: tf.types.experimental.ConcreteFunction,
) -> SignatureFlat:
  """Returns the flattened input signature of the given function."""
  leaves, tree_def = jax.tree.flatten(fn.structured_input_signature)
  input_names = [arg.name for arg in fn.function_def.signature.input_arg]
  if len(input_names) < len(leaves):
    raise ValueError(
        f'The number of input arguments in FunctionDef ({len(input_names)}) is'
        ' smaller than the number of leaves in the flattened input signature'
        f' ({len(leaves)})'
    )
  return SignatureFlat(input_names[: len(leaves)], leaves, tree_def)


def _unwrap_tf_trackables(
    tf_trackable: Any,
) -> Any:
  """Unwraps TF trackables from the given PyTree."""
  if isinstance(tf_trackable, Mapping):
    return {k: _unwrap_tf_trackables(v) for k, v in tf_trackable.items()}
  elif isinstance(tf_trackable, Sequence) and not isinstance(
      tf_trackable, (str, bytes)
  ):
    return [_unwrap_tf_trackables(v) for v in tf_trackable]
  else:
    return tf_trackable


def _output_name_for_key(key: Any) -> str:
  if isinstance(key, jax.tree_util.SequenceKey):
    return f'{_OUTPUT_NAME_PREFIX}_{key.idx}'
  elif isinstance(key, jax.tree_util.DictKey):
    return f'{key.key}'
  elif isinstance(key, jax.tree_util.GetAttrKey):
    return f'{key.name}'
  raise ValueError(f'Invalid output key type: {key}')


def _output_name(path: Sequence[Any]) -> str:
  """Returns the output name based on its path in the output signature."""
  if not path:
    return f'{_OUTPUT_NAME_PREFIX}_0'
  return '.'.join(_output_name_for_key(k) for k in path)


def _output_names(
    fn: tf.types.experimental.ConcreteFunction,
) -> Sequence[str]:
  """Returns the flattened output signature of the given function."""
  unwrapped_outputs = _unwrap_tf_trackables(fn.structured_outputs)
  leaves_with_path = jax.tree.leaves_with_path(unwrapped_outputs)
  if not leaves_with_path:
    return []
  paths, _ = zip(*leaves_with_path)
  return [_output_name(path) for path in paths]


def get_output_signature(
    fn: tf.types.experimental.ConcreteFunction,
) -> jaxtyping.PyTree[tf.TensorSpec]:
  """Returns the output signature of the TF function.

  Tensor names in the output signature match the output names of the TF function
  in the TF SavedModel.

  Args:
    fn: A concrete TF function.
  """
  output_names_iter = iter(list(_output_names(fn)))

  return jax.tree.map(
      lambda t: tf.TensorSpec(
          shape=t.shape, dtype=t.dtype, name=next(output_names_iter)
      ),
      fn.structured_outputs,
  )


def _to_keyword_only_fn(
    f: tf.types.experimental.ConcreteFunction,
) -> tf.types.experimental.ConcreteFunction:
  """Wraps a function into one whose inputs and outputs are keyword-only.

  Args:
    f: a TF concrete function.

  Returns:
    The wrapped function (also a TF concrete function).
  """
  input_names, input_leaves, input_def = _flat_input_signature(f)
  output_names = _output_names(f)

  if input_names is None and output_names is None:
    return f

  new_input_signature = _make_dict_only_signature(input_names, input_leaves)

  @tf.function(
      autograph=False,
  )
  def new_f(**input_dict):
    args, kwargs = _to_args_kwargs(
        _dict_to_tree(
            input_dict,
            input_names,
            input_def,
        )
    )
    output = f(*args, **kwargs)
    new_output = _tree_to_dict(
        output,
        output_names,
    )
    return new_output

  return new_f.get_concrete_function(
      **new_input_signature,
  )


def _add_defaults_to_signatures(
    saved_model_dir: str,
    fns: (
        Mapping[str, tf.types.experimental.ConcreteFunction | PreparedFunction]
        | None
    ),
) -> None:
  """Adds defaults to the signatures of the functions in TF SavedModel."""
  if not fns:
    return

  saved_model_pb_path = os.path.join(saved_model_dir, 'saved_model.pb')
  model_proto = saved_model_pb2.SavedModel()

  with file_utils.open_file(saved_model_pb_path, 'rb') as f:
    model_proto.ParseFromString(f.read())

  for name, fn in fns.items():
    if not isinstance(fn, PreparedFunction):
      continue
    if not fn.defaults:
      continue
    flattened_defaults = jax.tree.leaves(fn.defaults)
    input_names, _, _ = _flat_input_signature(fn.concrete_fn)
    tf_default_names = input_names[-len(flattened_defaults) :]
    sig_def = model_proto.meta_graphs[0].signature_def[name]

    for arg_name, default in zip(tf_default_names, flattened_defaults):
      sig_def.defaults[arg_name].CopyFrom(tf.make_tensor_proto(default))

  with file_utils.open_file(saved_model_pb_path, 'wb') as f:
    f.write(model_proto.SerializeToString())


def save_tf_functions(
    model_dir: str,
    fns: Mapping[
        str, tf.types.experimental.ConcreteFunction | PreparedFunction
    ],
    *,
    trackable_resources: Any = None,
    tf_saved_model_sub_dir: str = NPT_TF_SAVED_MODEL_SUB_DIR,
) -> dict[str, persistence_lib.GlobalSupplemental]:
  """Saves TensorFlow (TF) concrete functions with names to a TF SavedModel.

  The TF SavedModel will have enough information to let one retrieve (and call)
  the concrete functions by their names.

  Args:
    model_dir: The Neptune model directory.
    fns: The mapping from names to concrete functions.
    trackable_resources: The nested structure (i.e. PyTree) of
      `tf.saved_model.experimental.TrackableResource`s that are used in
      `concrete_functions`. All TF resources the concrete functions use
      (directly or indirectly) must be present in this structure. Otherwise, an
      "untracked resource" error will be raised. If tf.Module is passed, it will
      be used to create the saved model.
    tf_saved_model_sub_dir: Saved model dir name relative to `model_dir`.

  Returns:
    The single-entry dictionary with the resulting TF supplemental.
  """
  concrete_fns: MutableMapping[str, tf.types.experimental.ConcreteFunction] = {}

  for name, fn in fns.items():
    if isinstance(fn, PreparedFunction):
      concrete_fns[name] = fn.concrete_fn
    else:
      concrete_fns[name] = fn

  wrapped_fns = {k: _to_keyword_only_fn(v) for k, v in concrete_fns.items()}

  tf_module = tf.Module()
  if isinstance(trackable_resources, tf.Module):
    tf_module = trackable_resources
  elif trackable_resources is not None:
    tf_module.resources = trackable_resources

  target_path = os.path.join(model_dir, tf_saved_model_sub_dir)
  tf.saved_model.save(tf_module, target_path, signatures=wrapped_fns)

  _add_defaults_to_signatures(target_path, fns)

  return {
      TF_SAVED_MODEL_SUPPLEMENTAL_NAME: persistence_lib.GlobalSupplemental(
          tf_saved_model_as_npt_supplemental(tf_saved_model_sub_dir)
      )
  }
