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

"""Tools for loading the TensorFlow functions saved with a model."""

import collections
from collections.abc import Callable
import os
import pathlib
from typing import Any

import tensorflow as tf

from xla_saved_model._src.core import constants as core_constants

_FUNCTION_TYPES = (
    tf.types.experimental.PolymorphicFunction,
    tf.types.experimental.ConcreteFunction,
)


class TfFunction(tf.Module):
  """A function restored from a model's TF SavedModel, callable as the original.

  The restored SavedModel stays tracked, so a `tf.Module` or Keras model that
  calls the function also saves the variables and tables that it uses.
  """

  def __init__(self, saved_model: Any, function: Callable[..., Any]):
    super().__init__()
    self.saved_model = saved_model
    self._function = function

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    return self._function(*args, **kwargs)


def _find_functions(
    root: Any, function_name: str
) -> list[tuple[str, Callable[..., Any]]]:
  """Returns the path and value of every function named `function_name`.

  The root and every object that it tracks are searched, breadth first. Only
  `tf.function`s and concrete functions count, not other attributes of that
  name, such as a method of a restored table. A function found on more than one
  object is returned once.

  Args:
    root: The root of the object graph.
    function_name: The attribute name to look for.

  Returns:
    `(path, function)` pairs, in the order found.
  """
  found = []
  found_ids = set()
  visited_ids = {id(root)}
  queue: collections.deque[tuple[tuple[str, ...], Any]] = collections.deque(
      [((), root)]
  )
  while queue:
    path, node = queue.popleft()
    function = getattr(node, function_name, None)
    if isinstance(function, _FUNCTION_TYPES) and id(function) not in found_ids:
      found_ids.add(id(function))
      found.append(("/".join((*path, function_name)), function))
    for name, child in tf.train.TrackableView.children(node).items():
      if id(child) not in visited_ids:
        visited_ids.add(id(child))
        queue.append(((*path, name), child))
  return found


def _default_saved_model_sub_dir(model_dir: str) -> str:
  """Returns the TF SavedModel of a model that keeps its whole object graph.

  That is the unconverted copy if the model kept one. Otherwise, the TF
  SavedModel was either never converted, and so is unconverted itself, or was
  converted without keeping a copy, in which case it is the only one there is.

  Args:
    model_dir: Directory where the model was saved.
  """
  unconverted = core_constants.UNCONVERTED_TF_SAVED_MODEL_SUB_DIR
  if tf.io.gfile.isdir(os.path.join(model_dir, unconverted)):
    return unconverted
  return core_constants.TF_SAVED_MODEL_SUB_DIR


def load_tf_fn(
    model_dir: pathlib.Path | str,
    function_name: str,
    *,
    saved_model_sub_dir: str | None = None,
) -> TfFunction:
  """Loads a TensorFlow function saved with a model.

  The function is looked up by name on the objects of the model's TF
  SavedModel: its root and every object that the root tracks, so the function
  can be saved on any of them. If none has it, a signature of that name is used.

  By default the TF SavedModel is the one that keeps the object graph: the copy
  kept from before the TF SavedModel was converted for serving, if the model has
  one, and the TF SavedModel served otherwise. A converted TF SavedModel keeps
  only its signatures, so a function saved as an attribute can only be found in
  the unconverted one.

  Args:
    model_dir: Directory where the model was saved.
    function_name: The attribute name that the function was saved under, or a
      signature key.
    saved_model_sub_dir: Directory of the TF SavedModel, relative to
      `model_dir`. Defaults to the one that keeps the object graph, as above.

  Returns:
    The function.

  Raises:
    OSError: If there is no SavedModel in `saved_model_sub_dir`.
    ValueError: If neither an object nor a signature of the SavedModel has the
      function, or different objects have different functions of that name.
  """
  model_dir = str(model_dir)
  if saved_model_sub_dir is None:
    saved_model_sub_dir = _default_saved_model_sub_dir(model_dir)
  saved_model_dir = os.path.join(model_dir, saved_model_sub_dir)
  saved_model = tf.saved_model.load(saved_model_dir)

  found = _find_functions(saved_model, function_name)
  if len(found) > 1:
    raise ValueError(
        f"More than one function is named '{function_name}' in"
        f" {saved_model_dir}: {[path for path, _ in found]}."
    )
  if found:
    ((_, function),) = found
    return TfFunction(saved_model, function)

  signatures = getattr(saved_model, "signatures", {})
  if function_name in signatures:
    return TfFunction(saved_model, signatures[function_name])
  raise ValueError(
      f"No function is named '{function_name}' in {saved_model_dir}."
      f" Signatures: {sorted(signatures)}."
  )


__all__ = [
    "TfFunction",
    "load_tf_fn",
]
