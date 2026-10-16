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

import os
import pathlib

import tensorflow as tf

from xla_saved_model._src.core import constants as core_constants
from xla_saved_model.tools.stitching import tf_function_loader

_FUNCTION_NAME = 'lookup_and_scale'
_KEYS_SPEC = tf.TensorSpec([None], tf.string)


class _Graph(tf.Module):
  """Holds a function that reads a variable and a lookup table."""

  def __init__(self):
    super().__init__()
    self.scale = tf.Variable(2.0)
    self.table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(['a', 'b'], [1.0, 10.0]),
        default_value=0.0,
    )

  @tf.function(input_signature=[_KEYS_SPEC])
  def lookup_and_scale(self, keys):
    return self.table.lookup(keys) * self.scale


class _Caller(tf.Module):
  """Calls a loaded function, as a model stitching it would."""

  def __init__(self, function):
    super().__init__()
    self.function = function

  @tf.function(input_signature=[_KEYS_SPEC])
  def serve(self, keys):
    return self.function(keys) + 1.0


def _save(root, model_dir, sub_dir='tf_saved_model', **kwargs):
  tf.saved_model.save(root, os.path.join(model_dir, sub_dir), **kwargs)


def _frontend(*graphs):
  """Returns a root whose functions sit two objects below its resources."""
  frontend = tf.Module()
  frontend.graphs = list(graphs)
  root = tf.Module()
  root.resources = [frontend]
  return root


class TfFunctionLoaderTest(tf.test.TestCase):

  def test_finds_a_nested_function(self):
    model_dir = self.create_tempdir().full_path
    _save(_frontend(_Graph()), model_dir)

    function = tf_function_loader.load_tf_fn(model_dir, _FUNCTION_NAME)

    self.assertIsInstance(function, tf_function_loader.TfFunction)
    self.assertAllClose(function(tf.constant(['a', 'b', 'c'])), [2, 20, 0])

  def test_prefers_the_unconverted_saved_model(self):
    model_dir = self.create_tempdir().full_path
    # The TF SavedModel served keeps only a signature, as a converted one does,
    # and the unconverted copy keeps the object graph with the function.
    graph = _Graph()
    _save(
        graph,
        model_dir,
        signatures={
            'lookup': graph.lookup_and_scale.get_concrete_function(_KEYS_SPEC)
        },
    )
    _save(
        _frontend(_Graph()),
        model_dir,
        sub_dir=core_constants.UNCONVERTED_TF_SAVED_MODEL_SUB_DIR,
    )

    function = tf_function_loader.load_tf_fn(model_dir, _FUNCTION_NAME)

    self.assertAllClose(function(tf.constant(['a', 'b'])), [2, 20])
    # The signature of the TF SavedModel served is not in the unconverted copy.
    with self.assertRaisesRegex(ValueError, "No function is named 'lookup'"):
      tf_function_loader.load_tf_fn(model_dir, 'lookup')

  def test_loads_from_a_saved_model_sub_dir(self):
    model_dir = self.create_tempdir().full_path
    _save(_frontend(_Graph()), model_dir, sub_dir='unconverted')

    function = tf_function_loader.load_tf_fn(
        pathlib.Path(model_dir),
        _FUNCTION_NAME,
        saved_model_sub_dir='unconverted',
    )

    self.assertAllClose(function(tf.constant(['b'])), [20])

  def test_a_model_calling_the_function_can_be_saved(self):
    model_dir = self.create_tempdir().full_path
    _save(_frontend(_Graph()), model_dir)
    function = tf_function_loader.load_tf_fn(model_dir, _FUNCTION_NAME)

    caller_dir = self.create_tempdir().full_path
    _save(_Caller(function), caller_dir)

    caller = tf.saved_model.load(os.path.join(caller_dir, 'tf_saved_model'))
    self.assertAllClose(caller.serve(tf.constant(['a', 'c'])), [3, 1])

  def test_a_function_on_several_objects_is_not_ambiguous(self):
    graph = _Graph()
    root = _frontend(graph)
    root.lookup_and_scale = graph.lookup_and_scale
    model_dir = self.create_tempdir().full_path
    _save(root, model_dir)

    function = tf_function_loader.load_tf_fn(model_dir, _FUNCTION_NAME)

    self.assertAllClose(function(tf.constant(['a'])), [2])

  def test_different_functions_of_the_same_name_raise(self):
    model_dir = self.create_tempdir().full_path
    _save(_frontend(_Graph(), _Graph()), model_dir)

    with self.assertRaisesRegex(
        ValueError, 'resources/0/graphs/0/.*resources/0/graphs/1/'
    ):
      tf_function_loader.load_tf_fn(model_dir, _FUNCTION_NAME)

  def test_falls_back_to_a_signature(self):
    graph = _Graph()
    model_dir = self.create_tempdir().full_path
    _save(
        graph,
        model_dir,
        signatures={
            'lookup': graph.lookup_and_scale.get_concrete_function(_KEYS_SPEC)
        },
    )

    function = tf_function_loader.load_tf_fn(model_dir, 'lookup')

    self.assertAllClose(function(keys=tf.constant(['a']))['output_0'], [2])

  def test_missing_function_raises(self):
    model_dir = self.create_tempdir().full_path
    _save(_frontend(_Graph()), model_dir)

    with self.assertRaisesRegex(ValueError, "No function is named 'missing'"):
      tf_function_loader.load_tf_fn(model_dir, 'missing')

  def test_missing_saved_model_raises(self):
    with self.assertRaises(OSError):
      tf_function_loader.load_tf_fn(
          self.create_tempdir().full_path, _FUNCTION_NAME
      )


if __name__ == '__main__':
  tf.test.main()
