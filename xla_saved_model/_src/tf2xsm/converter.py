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

"""Converts TF functions to XSE functions (allowing TF resources)."""

from collections.abc import Callable, Mapping, MutableMapping, Sequence
import copy
import dataclasses
import functools
import inspect
import os
from typing import Any, cast

import jax
import jaxtyping
import tensorflow as tf

from xla_saved_model.protos import manifest_pb2
from xla_saved_model.protos import tf_concrete_function_handle_pb2
from xla_saved_model._src.core import file_utils
from xla_saved_model._src.core import persistence_lib
from xla_saved_model._src.core import serializable_function
from xla_saved_model._src.core import shlo_type
from xla_saved_model._src.core import unstructured_data
from xla_saved_model._src.tf2xsm import type_conversion

# pylint: disable=g-direct-tensorflow-import
from tensorflow.core.protobuf import saved_model_pb2

# pylint: enable=g-direct-tensorflow-import

TF_CONCRETE_FUNCTION_HANDLE_MIME_TYPE = (
    'application/protobuf;'
    ' type=xla_saved_model_tf_concrete_function_handle.TfConcreteFunctionHandle'
)
TF_CONCRETE_FUNCTION_HANDLE_VERSION = '0.0.1'
XSM_TF_SAVED_MODEL_SUB_DIR = 'tf_saved_model/'
TF_SAVED_MODEL_SUPPLEMENTAL_NAME = 'tensorflow_saved_model'
SAVED_MODEL_MIME_TYPE = 'application/x.tensorflow-saved-model'
SAVED_MODEL_VERSION = '1.0'

_OUTPUT_NAME_PREFIX = 'output'
_PB_EXT_NAME = 'pb'


@dataclasses.dataclass(frozen=True)
class PreparedFunction:
  """TF function prepared for XSE export.

  Attributes:
    concrete_fn: The concrete TF function (to be saved in the TF SavedModel).
    serializable_fn: The XSE function generated from `concrete_fn`.
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


def _xsm_name(spec: tf.TypeSpec, path: tuple[Any, ...]) -> str | None:
  """Returns the name of the TF spec to be used in the XSE signature."""
  if hasattr(spec, 'name') and spec.name is not None:
    return spec.name

  if not path:
    return None

  path_key = path[-1]

  if hasattr(path_key, 'key'):
    return str(path_key.key)

  if hasattr(path_key, 'name'):
    return str(path_key.name)

  return None


def _xsm_signature_from_tf_signature(
    tf_signature: jaxtyping.PyTree[tf.TypeSpec],
) -> jaxtyping.PyTree[shlo_type.ShloTensorSpec]:
  """Converts a TF signature to an XSE signature."""

  def _convert_leaf(path, leaf):
    xsm_name = _xsm_name(leaf, path)

    shlo_spec = type_conversion.shlo_tensor_spec_from(leaf)
    if isinstance(shlo_spec, tuple):
      # The value has been expanded from a composite type; we keep TF-like
      # naming convention for the flattened arguments, e.g. for a composite
      # input named 'x' we will have names 'x', 'x_1', etc. for the leaves.
      for i, spec in enumerate(shlo_spec):
        spec.name = xsm_name if i == 0 else f'{xsm_name}_{i}'
    else:
      shlo_spec.name = xsm_name

    return shlo_spec

  return jax.tree.map_with_path(_convert_leaf, tf_signature)


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
  tf_input_signature = jax.tree.map(
      type_conversion.tf_type_spec_from, input_signature
  )
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
  """Converts the TF concrete function to an XSE function.

  The resulting XSE function is effectively a reference to a function
  stored in TF SavedModel by `save_tf_functions`.

  Args:
    fn_name: The name to be used in the XSE manifest to refer to the TF
      function.
    fn: The function to prepare.
    input_signature: The input signature to use for the TF function. If None,
      the input signature will be inferred from the function.
    suppress_x64_outputs: If True, casts 64-bit outputs to 32-bit.

  Returns:
    The XSE function referring to the original TF function in the TF
    SavedModel.
  """

  def _set_xsm_name(path, x):
    if isinstance(x, tf.TensorSpec):
      return tf.TensorSpec(
          shape=x.shape, dtype=x.dtype, name=_xsm_name(x, path)
      )
    return x

  input_signature_with_names = jax.tree.map_with_path(
      _set_xsm_name, input_signature
  )

  initial_concrete_fn, defaults = _get_concrete_function(
      fn, input_signature_with_names
  )

  if suppress_x64_outputs and _has_x64_outputs(initial_concrete_fn):
    concrete_fn, _ = _get_concrete_function(
        _suppress_x64_outputs(fn),
        initial_concrete_fn.structured_input_signature,
    )
  else:
    concrete_fn = initial_concrete_fn

  input_names, _ = _flat_input_signature(concrete_fn)
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
          input_signature=_xsm_signature_from_tf_signature(
              final_input_signature
          ),
          output_signature=_xsm_signature_from_tf_signature(
              final_output_signature
          ),
      ),
      concrete_fn=concrete_fn,
      defaults=defaults,
  )


def tf_saved_model_as_xsm_supplemental(
    subdir: str,
) -> manifest_pb2.UnstructuredData:
  """An XLA SavedModel global supplemental pointing to a TF SavedModel (path).

  Args:
    subdir: the path to the TF SavedModel dir. This can be a path relative to
      the XLA SavedModel dir, such as 'tf_saved_model/', or an absolute path.

  Returns:
    An `manifest_pb2.UnstructuredData` which can be used as an XLA SavedModel
    global supplemental.
  """
  return manifest_pb2.UnstructuredData(
      file_system_location=manifest_pb2.FileSystemLocation(string_path=subdir),
      mime_type=SAVED_MODEL_MIME_TYPE,
      version=SAVED_MODEL_VERSION,
  )


def _make_dict_only_signature(
    names: Sequence[str], leaves: Sequence[tf.TypeSpec]
) -> Mapping[str, tf.TypeSpec]:
  """Creates a dict-only signature from names and leaves.

  Args:
    names: A sequence of names for the leaves.
    leaves: A sequence of TensorFlow value types (e.g. TensorSpec,
      RaggedTensorSpec).

  Returns:
    A dict-only signature where keys are names and values are tensor value
    types.
  """

  def _clear_name(v):
    if not isinstance(v, tf.TensorSpec):
      return v
    return tf.TensorSpec(shape=v.shape, dtype=v.dtype, name=None)

  return {k: _clear_name(v) for k, v in zip(names, leaves)}


def _tree_to_dict(
    tree: jaxtyping.PyTree[tf.TypeSpec],
    names: Sequence[str],
) -> Mapping[str, tf.TensorSpec]:
  """Converts a tree of TF value types to a dictionary of TensorSpecs."""
  flat_values = []
  for leaf in jax.tree.leaves(tree):
    # If the leaf is a composite TF type, expand it to its flat TensorSpecs.
    flat_values.extend(tf.nest.flatten(leaf, expand_composites=True))
  return dict(zip(names, flat_values))


def _dict_to_tree(
    d: Mapping[str, tf.TensorSpec],
    names: Sequence[str],
    tree_structure: Any,
) -> jaxtyping.PyTree[tf.TypeSpec]:
  """Converts a dictionary of TensorSpecs to a tree of TF value types."""
  names_iter = iter(names)

  def _map_leaf(leaf):
    flat_leaf = tf.nest.flatten(leaf, expand_composites=True)
    packed_leaf = [d[next(names_iter)] for _ in flat_leaf]
    return tf.nest.pack_sequence_as(leaf, packed_leaf, expand_composites=True)

  return jax.tree.map(_map_leaf, tree_structure)


def _flat_input_signature(
    fn: tf.types.experimental.ConcreteFunction,
) -> tuple[Sequence[str], Sequence[tf.TensorSpec]]:
  """Returns the flattened input signature of the given function."""
  leaves = []
  # We use JAX flattening here to suppress handling None values
  # which tf.nest.flatten would keep. For each leaf, we potentially
  # expand it to its flat TensorSpecs if it is a composite TF type.
  for leaf in jax.tree.leaves(fn.structured_input_signature):
    leaves.extend(tf.nest.flatten(leaf, expand_composites=True))
  input_names = [arg.name for arg in fn.function_def.signature.input_arg]
  if len(input_names) < len(leaves):
    raise ValueError(
        f'The number of input arguments in FunctionDef ({len(input_names)}) is'
        ' smaller than the number of leaves in the flattened input signature'
        f' ({len(leaves)})'
    )
  return (input_names[: len(leaves)], leaves)


# TF replaces a container owned by a trackable (a dict attribute of a
# `tf.Module`, the outputs of a function restored from a SavedModel, ...) with a
# wrapper such as `_DictWrapper`, `ListWrapper` or `_TupleWrapper`. They all
# derive from this class.
_TF_CONTAINER_WRAPPER = tf.__internal__.tracking.TrackableDataStructure


def _is_tf_container_wrapper(node: Any) -> bool:
  return isinstance(node, _TF_CONTAINER_WRAPPER)


def _rebuild_tf_container_wrapper(wrapper: Any) -> Any:
  """Rebuilds a TF container wrapper as the builtin container it wraps."""
  if isinstance(wrapper, Mapping):
    return {k: _unwrap_tf_trackables(v) for k, v in wrapper.items()}
  values = [_unwrap_tf_trackables(v) for v in wrapper]
  if hasattr(wrapper, '_fields'):  # A `_TupleWrapper` around a named tuple.
    # `type()` would return the wrapper; `__class__` is forwarded to the
    # wrapped named tuple.
    return wrapper.__class__(*values)
  return tuple(values) if isinstance(wrapper, tuple) else values


def _unwrap_tf_trackables(tree: Any) -> Any:
  """Replaces TF container wrappers in `tree` with builtin containers.

  Whether JAX can traverse a TF wrapper, and whether it then sees the wrapper's
  keys, depends on what has been registered with JAX in the process: it may be
  an opaque leaf, or a node whose children are identified by position only.
  Either way the structure and the names derived from it would be lost, so the
  wrappers are identified by type and rebuilt before JAX looks at the tree.

  Every other node is left as it is, including its type: a `chex.dataclass` or
  a named tuple is not a wrapper, and callers match their JAX-side structures
  (e.g. `input_polymorphic_shapes`) against it.

  Args:
    tree: A PyTree that may contain TF container wrappers.

  Returns:
    The same tree with TF container wrappers replaced by builtin containers.
  """
  return jax.tree.map(
      lambda node: (
          _rebuild_tf_container_wrapper(node)
          if _is_tf_container_wrapper(node)
          else node
      ),
      tree,
      is_leaf=_is_tf_container_wrapper,
  )


def _output_name_for_key(key: Any) -> str:
  if isinstance(key, jax.tree_util.SequenceKey):
    return f'{_OUTPUT_NAME_PREFIX}_{key.idx}'
  elif isinstance(key, jax.tree_util.DictKey):
    return f'{key.key}'
  elif isinstance(key, jax.tree_util.GetAttrKey):
    return f'{key.name}'
  raise ValueError(
      f'Cannot derive an output name from JAX key {key!r} (type'
      f' {type(key).__name__}). Output names come from the keys JAX assigns'
      ' to each node in the output structure; a node registered with'
      ' `jax.tree_util.register_pytree_node` has positional keys only.'
      ' Register the output type with'
      ' `jax.tree_util.register_pytree_with_keys` or return a dict, list,'
      ' tuple, named tuple or dataclass instead.'
  )


def _output_name(path: Sequence[Any]) -> str:
  """Returns the output name based on its path in the output signature."""
  if not path:
    return f'{_OUTPUT_NAME_PREFIX}_0'
  return '.'.join(_output_name_for_key(k) for k in path)


def _component_output_name(base_name: str, index: int) -> str:
  """Returns the output name of the `index`-th tensor of a leaf.

  A composite TF type (e.g. a `tf.RaggedTensor`) is one leaf but holds several
  tensors, and XSM has no composite type, so each component needs a name.

  Args:
    base_name: The output name of the leaf.
    index: The index of the tensor in the flattened leaf.
  """
  return base_name if index == 0 else f'{base_name}_{index - 1}'


def _output_names(
    fn: tf.types.experimental.ConcreteFunction,
) -> Sequence[str]:
  """Returns the flattened output signature of the given function."""
  names = []
  for path, leaf in jax.tree.leaves_with_path(
      _unwrap_tf_trackables(fn.structured_outputs)
  ):
    base_name = _output_name(path)
    num_tensors = len(tf.nest.flatten(leaf, expand_composites=True))
    names.extend(
        _component_output_name(base_name, i) for i in range(num_tensors)
    )
  return names


def get_output_signature(
    fn: tf.types.experimental.ConcreteFunction,
) -> jaxtyping.PyTree[tf.TensorSpec]:
  """Returns the output signature of the TF function.

  Tensor names in the output signature match the output names of the TF function
  in the TF SavedModel.

  Args:
    fn: A concrete TF function.
  """

  def _map_leaf(path, leaf):
    base_name = _output_name(path)
    flat_specs = tuple(
        tf.TensorSpec(
            shape=s.shape,
            dtype=s.dtype,
            name=_component_output_name(base_name, i),
        )
        for i, s in enumerate(tf.nest.flatten(leaf, expand_composites=True))
    )
    return flat_specs[0] if len(flat_specs) == 1 else flat_specs

  return jax.tree.map_with_path(
      _map_leaf, _unwrap_tf_trackables(fn.structured_outputs)
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
  input_names, input_leaves = _flat_input_signature(f)
  output_names = _output_names(f)

  new_input_signature = _make_dict_only_signature(input_names, input_leaves)

  @tf.function(
      autograph=False,
  )
  def new_f(**input_dict):
    args, kwargs = _to_args_kwargs(
        _dict_to_tree(
            input_dict,
            input_names,
            f.structured_input_signature,
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
    input_names, _ = _flat_input_signature(fn.concrete_fn)
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
    save_options: tf.saved_model.SaveOptions | None = None,
    tf_saved_model_sub_dir: str = XSM_TF_SAVED_MODEL_SUB_DIR,
) -> dict[str, persistence_lib.GlobalSupplemental]:
  """Saves TensorFlow (TF) concrete functions with names to a TF SavedModel.

  The TF SavedModel will have enough information to let one retrieve (and call)
  the concrete functions by their names.

  Args:
    model_dir: The XLA SavedModel directory.
    fns: The mapping from names to concrete functions.
    trackable_resources: The nested structure (i.e. PyTree) of
      `tf.saved_model.experimental.TrackableResource`s that are used in
      `concrete_functions`. All TF resources the concrete functions use
      (directly or indirectly) must be present in this structure. Otherwise, an
      "untracked resource" error will be raised. If tf.Module is passed, it will
      be used to create the saved model.
    save_options: An optional `tf.saved_model.SaveOptions` for configuring save
      options.
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
  tf.saved_model.save(
      tf_module, target_path, signatures=wrapped_fns, options=save_options
  )

  _add_defaults_to_signatures(target_path, fns)

  return {
      TF_SAVED_MODEL_SUPPLEMENTAL_NAME: persistence_lib.GlobalSupplemental(
          tf_saved_model_as_xsm_supplemental(tf_saved_model_sub_dir)
      )
  }
