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

"""XLA SavedModel export example with CompositeProcessor."""

import os
import tempfile

import jax
import jax.numpy as jnp

from xla_serving_data.resource_manager.python import resource_manager
from xla_serving_data_test.examples.composite_processor import add_int_from_resource_transform_pybind
from xla_serving_data_test.examples.composite_processor import add_two_input_buffers_transform_pybind
from xla_serving_data_test.examples.composite_processor import int_resource_pybind
from xla_saved_model.protos import orchestration_pb2
from xla_saved_model._src.core import shlo_type
from xla_saved_model._src.export import jax_model
from xla_saved_model._src.export import main_lib
from xla_saved_model._src.export import serving_config


class AddTwoInputBuffersModule:
  """A transform module for adding two host buffers."""

  def __init__(self):
    self._transform = (
        add_two_input_buffers_transform_pybind.AddTwoInputBuffersTransform()
    )

  @property
  def transform(self):
    return self._transform

  def export_assets(self) -> dict[str, str]:
    return {}

  def get_output_signature(self, input_signature):
    return {'x': input_signature['x']}


class AddFromResourceModule:
  """A transform module adding a value from an IntResource."""

  def __init__(self, resource: int_resource_pybind.ExamplesIntResource):
    self._transform = (
        add_int_from_resource_transform_pybind.AddIntFromResourceTransform(
            resource
        )
    )

  @property
  def transform(self):
    return self._transform

  def export_assets(self) -> dict[str, str]:
    return self.transform.assets()

  def get_output_signature(self, input_signature):
    return input_signature


def _create_int_resource(
    path: str,
    name: str,
    value: int,
) -> int_resource_pybind.ExamplesIntResource:
  os.makedirs(os.path.dirname(path))
  with open(path, 'w') as f:
    f.write(str(value))
  asset = resource_manager.Asset(path=path, name=name)
  return int_resource_pybind.ExamplesIntResource([asset])


def export_model(export_path: str):
  """Exports a dummy model with a CompositeProcessor managing multiple resources."""

  def apply_fn(params, inputs):
    del params
    return {'pred': jax.tree.map(lambda x: x * 2, inputs)['x']}

  input_signature = {
      'x': shlo_type.ShloTensorSpec(
          shape=(None, 4), dtype=shlo_type.ShloDType.i32
      ),
      'y': shlo_type.ShloTensorSpec(
          shape=(None, 4), dtype=shlo_type.ShloDType.i32
      ),
  }

  resource_dir = tempfile.mkdtemp()
  resource1 = _create_int_resource(
      path=os.path.join(resource_dir, 'dir1', 'data.txt'),
      name='resource1',
      value=5,
  )
  resource2 = _create_int_resource(
      path=os.path.join(resource_dir, 'dir2', 'data.txt'),
      name='resource2',
      value=10,
  )
  resource3 = _create_int_resource(
      path=os.path.join(resource_dir, 'dir3', 'data.txt'),
      name='resource3',
      value=8,
  )

  t1 = AddTwoInputBuffersModule()
  t2 = AddFromResourceModule(resource1)
  t3 = AddFromResourceModule(resource2)
  t4 = AddFromResourceModule(resource3)

  composite_preprocessor = main_lib.CompositeModel(
      name='transform_inputs',
      transforms=[t1, t2, t3],
  )

  composite_postprocessor = main_lib.CompositeModel(
      name='postprocess_output',
      transforms=[t4],
  )

  jm = jax_model.JaxModel(
      apply_fn_bundle=jax_model.ApplyFnBundle(
          apply_fn=apply_fn,
          input_polymorphic_shapes={'x': 'b, _'},
          jax_export_kwargs={'platforms': ['cpu']},
      ),
      params={
          'w': jnp.ones((1, 1)),
          'b': jnp.zeros((1,)),
      },
      name='jax_model',
  )

  main_lib.save(
      [composite_preprocessor, jm, composite_postprocessor],
      export_path=export_path,
      serving_configs=[
          serving_config.ServingConfig(
              serving_signature='serving_default',
              input_signature=input_signature,
              orchestration=serving_config.Orchestration(
                  model_calls=[
                      serving_config.ModelCall(
                          model_name='transform_inputs',
                          method_key='serving_default',
                          role=orchestration_pb2.ROLE_PREPROCESSOR,
                      ),
                      serving_config.ModelCall(
                          model_name='jax_model',
                          method_key='serving_default',
                          role=orchestration_pb2.ROLE_MODEL,
                      ),
                      serving_config.ModelCall(
                          model_name='postprocess_output',
                          method_key='serving_default',
                          role=orchestration_pb2.ROLE_POSTPROCESSOR,
                      ),
                  ]
              ),
          )
      ],
  )
