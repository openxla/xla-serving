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

"""Tests for neptune.model._src.export.main_lib."""

import pathlib

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp

from neptune_model._src import assets2npt
from neptune_model._src.core import persistence_lib
from neptune_model._src.core import test_utils
from neptune_model._src.core import value
from neptune_model._src.export import jax_model
from neptune_model._src.export import main_lib
from neptune_model._src.export import serving_config
from neptune_model._src.processors import processor
from neptune_model.protos import asset_map_pb2
from neptune_model.protos import manifest_pb2
from neptune_model.protos import orchestration_pb2


class SaveModelTest(test_utils.NptTestCase):

  def test_save_model_with_one_config(self):
    """Tests that save() runs."""
    model = jax_model.JaxModel(
        apply_fn_bundle=jax_model.ApplyFnBundle(
            apply_fn=lambda params, inputs: {'out': inputs['x'] * params['w']},
            jax_export_kwargs={'platforms': ['cpu']},
        ),
        params={'w': jnp.array([2.0], dtype=jnp.float32)},
    )
    batch_options = orchestration_pb2.BatchOptions(
        batch_component=orchestration_pb2.BatchOptions.MODEL_FUNCTION,
        max_batch_size=8,
        allowed_batch_sizes=[2, 4, 8],
        batch_timeout_micros=1000,
        max_enqueued_batches=10,
        num_batch_threads=2,
    )
    cfg = serving_config.ServingConfig(
        serving_signature='serving_default',
        input_signature={
            'x': jax.ShapeDtypeStruct(shape=(4,), dtype=jnp.float32)
        },
        batch_options=batch_options,
    )
    export_path = self.create_tempdir().full_path
    main_lib.save(
        model=model,
        export_path=export_path,
        serving_configs=[cfg],
    )

    manifest_proto = persistence_lib.load(export_path)
    orchestration_filename = manifest_proto.supplemental_info[
        'orchestration'
    ].file_system_location.string_path
    orchestration_path = pathlib.Path(export_path) / orchestration_filename
    pipelines_proto = orchestration_pb2.Pipelines.FromString(
        orchestration_path.read_bytes()
    )

    with self.subTest(name='validate_manifest_proto'):
      expected_manifest_proto = """
        objects {
          key: "serving_default"
          value {
            function {
              signature {
                input {
                  tuple {
                    elements {
                      tuple {
                        elements {
                          dict {
                            string_to_type {
                              key: "w"
                              value {
                                leaf {
                                  tensor_type {
                                    shape {
                                      shape_with_known_rank {
                                        dimension_sizes {
                                          size: 1
                                        }
                                      }
                                    }
                                    dtype: f32
                                  }
                                }
                              }
                            }
                          }
                        }
                        elements {
                          dict {
                            string_to_type {
                              key: "x"
                              value {
                                leaf {
                                  tensor_type {
                                    shape {
                                      shape_with_known_rank {
                                        dimension_sizes {
                                          size: 4
                                        }
                                      }
                                    }
                                    dtype: f32
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                    elements {
                      dict {
                      }
                    }
                  }
                }
                output {
                  dict {
                    string_to_type {
                      key: "out"
                      value {
                        leaf {
                          tensor_type {
                            shape {
                              shape_with_known_rank {
                                dimension_sizes {
                                  size: 4
                                }
                              }
                            }
                            dtype: f32
                          }
                        }
                      }
                    }
                  }
                }
              }
              body {
                stable_hlo_body {
                  calling_convention_version: 10
                  lowering_platforms: "cpu"
                  module_kept_var_idx: 0
                  module_kept_var_idx: 1
                }
              }
              data_names: "w"
              visibility: PUBLIC
            }
          }
        }
        objects {
          key: "weights"
          value {
            value {
              external {
                data {
                  file_system_location {
                    string_path: "checkpoint"
                  }
                  mime_type: "application/x.orbax-checkpoint"
                }
              }
            }
          }
        }
        supplemental_info {
          key: "orchestration"
          value {
            file_system_location {
              string_path: "orchestration.pb"
            }
            mime_type: "application/protobuf; type=neptune_orchestration.Pipelines"
            version: "0.0.1"
          }
        }
      """
      self.assertProtosEqual(
          expected_manifest_proto,
          manifest_proto,
          ignored_fields=[
              'objects.function.body.stable_hlo_body.stable_hlo',
              'objects.function.body.stable_hlo_body.supplemental_info',
          ],
      )

    with self.subTest(name='validate_orchestration_proto'):
      expected_pipelines_proto = """
        name_to_pipeline {
          key: "serving_default"
          value {
            signature {
              input {
                tuple {
                  elements {
                    tuple {
                      elements {
                        dict {
                          string_to_type {
                            key: "x"
                            value {
                              leaf {
                                tensor_type {
                                  shape {
                                    shape_with_known_rank {
                                      dimension_sizes {
                                        size: 4
                                      }
                                    }
                                  }
                                  dtype: f32
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                  elements {
                    dict {
                    }
                  }
                }
              }
              output {
                dict {
                  string_to_type {
                    key: "out"
                    value {
                      leaf {
                        tensor_type {
                          shape {
                            shape_with_known_rank {
                              dimension_sizes {
                                size: 4
                              }
                            }
                          }
                          dtype: f32
                        }
                      }
                    }
                  }
                }
              }
            }
            named_signature {
              inputs {
                named_tensor_types {
                  name: "x"
                  tensor_type {
                    shape {
                      shape_with_known_rank {
                        dimension_sizes {
                          size: 4
                        }
                      }
                    }
                    dtype: f32
                  }
                }
              }
              outputs {
                named_tensor_types {
                  name: "out"
                  tensor_type {
                    shape {
                      shape_with_known_rank {
                        dimension_sizes {
                          size: 4
                        }
                      }
                    }
                    dtype: f32
                  }
                }
              }
            }
            components {
              role: ROLE_MODEL
              function_name: "serving_default"
              weights_name: "weights"
            }
            batch_options {
              batch_component: MODEL_FUNCTION
              max_batch_size: 8
              batch_timeout_micros: 1000
              allowed_batch_sizes: 2
              allowed_batch_sizes: 4
              allowed_batch_sizes: 8
              num_batch_threads: 2
              max_enqueued_batches: 10
            }
          }
        }
      """
      self.assertProtosEqual(
          expected_pipelines_proto,
          pipelines_proto,
      )

    with self.subTest(name='checkpoint_exists'):
      self.assertTrue((pathlib.Path(export_path) / 'checkpoint').is_dir())

  def test_save_model_with_concretization(self):
    """Tests that save() works with concretized polymorphic shapes."""
    model = jax_model.JaxModel(
        apply_fn_bundle=jax_model.ApplyFnBundle(
            apply_fn=lambda params, x: {'out': x['input'] * params['w']},
            input_polymorphic_shapes={'input': 'b,_'},
            input_polymorphic_shape_symbol_values={'b': [1, 2]},
            jax_export_kwargs={'platforms': ['cpu']},
        ),
        params={'w': jnp.array([2.0], dtype=jnp.float32)},
    )
    cfg = serving_config.ServingConfig(
        serving_signature='serving_default',
        input_signature={
            'input': jax.ShapeDtypeStruct(shape=('b', 1), dtype=jnp.float32)
        },
    )
    export_path = self.create_tempdir().full_path
    main_lib.save(
        model=model,
        export_path=export_path,
        serving_configs=[cfg],
    )

    manifest_proto = persistence_lib.load(export_path)
    self.assertIn('serving_default', manifest_proto.objects)

    with self.subTest(name='validate_manifest_proto_concretization'):
      # When multiple functions are saved for a signature, they are stored
      # as a poly_fn with concrete_functions in the manifest.
      obj = manifest_proto.objects['serving_default']
      self.assertTrue(obj.HasField('poly_fn'))
      self.assertLen(obj.poly_fn.concrete_functions, 2)
      # Check the signature of one of the concrete functions (e.g., b=1).
      # Input is ((params, inputs), kwargs)
      input_tuple = obj.poly_fn.concrete_functions[0].signature.input.tuple
      # Params (first element of first tuple)
      self.assertEqual(
          input_tuple.elements[0]
          .tuple.elements[0]
          .dict.string_to_type['w']
          .leaf.tensor_type.shape.shape_with_known_rank.dimension_sizes[0]
          .size,
          1,
      )
      # Inputs (second element of first tuple)
      # The first dimension of the input should be 1 for the first concrete
      # function.
      self.assertEqual(
          input_tuple.elements[0]
          .tuple.elements[1]
          .dict.string_to_type['input']
          .leaf.tensor_type.shape.shape_with_known_rank.dimension_sizes[0]
          .size,
          1,
      )

    orchestration_filename = manifest_proto.supplemental_info[
        'orchestration'
    ].file_system_location.string_path
    orchestration_path = pathlib.Path(export_path) / orchestration_filename
    pipelines_proto = orchestration_pb2.Pipelines.FromString(
        orchestration_path.read_bytes()
    )

    with self.subTest(name='validate_orchestration_proto_polymorphism'):
      pipeline = pipelines_proto.name_to_pipeline['serving_default']
      # The first dimension of the first input should be dynamic
      # (size 0 in proto for None)
      input_type = (
          pipeline.signature.input.tuple.elements[0]
          .tuple.elements[0]
          .dict.string_to_type['input']
          .leaf.tensor_type
      )
      self.assertTrue(input_type.shape.HasField('shape_with_known_rank'))
      # In this proto representation, a dynamic dimension often has no size set.
      self.assertFalse(
          input_type.shape.shape_with_known_rank.dimension_sizes[0].HasField(
              'size'
          )
      )

  def test_save_model_with_assets(self):
    """Tests that on_persist_result returning either a supplemental or a value works."""
    src_asset_dir = self.create_tempdir('src_assets').full_path
    src_asset = pathlib.Path(src_asset_dir) / 'asset1.txt'
    src_asset.write_text('asset_content')

    def save_test_values(export_path: str):
      result = {
          'my_value': value.ExternalValue(
              data=manifest_pb2.UnstructuredData(
                  mime_type='test/value',
                  inlined_string='value_content',
              )
          ),
      }
      result.update(
          assets2npt.save_assets('my_supp', [str(src_asset)], export_path)
      )
      return result

    class PreparedTestProcessor:

      def __init__(self, input_signature, output_signature):
        self._input_signature = input_signature
        self._output_signature = output_signature

      @property
      def name(self) -> str:
        return 'test_processor'

      @property
      def input_signature(self):
        return self._input_signature

      @property
      def output_signature(self):
        return self._output_signature

      @property
      def weights_name(self) -> str | None:
        return None

      def to_npt_saveable(self):
        return value.ExternalValue(
            data=manifest_pb2.UnstructuredData(mime_type='mock')
        )

      def on_persist(self):
        return save_test_values

    class TestProcessor:

      def prepare(self, input_signature, **kwargs):
        del kwargs
        return PreparedTestProcessor(input_signature, input_signature)

    model = jax_model.JaxModel(
        apply_fn_bundle=jax_model.ApplyFnBundle(
            apply_fn=lambda params, inputs: {'out': inputs['x'] * params['w']},
            jax_export_kwargs={'platforms': ['cpu']},
        ),
        params={'w': jnp.array([2.0], dtype=jnp.float32)},
    )
    cfg = serving_config.ServingConfig(
        serving_signature='serving_default',
        input_signature={
            'x': jax.ShapeDtypeStruct(shape=(4,), dtype=jnp.float32)
        },
        processing=processor.SequentialProcessing(
            preprocessors=[TestProcessor()],
            postprocessors=[],
        ),
    )
    export_path = self.create_tempdir().full_path
    main_lib.save(
        model=model,
        export_path=export_path,
        serving_configs=[cfg],
    )

    manifest_proto = persistence_lib.load(export_path)

    with self.subTest(name='validate_assets'):
      self.assertIn('my_supp', manifest_proto.supplemental_info)
      supp = manifest_proto.supplemental_info['my_supp']
      self.assertEqual(
          supp.mime_type,
          'application/protobuf; type=neptune_assets.AssetMap',
      )
      self.assertEqual(
          supp.file_system_location.string_path, 'my_supp_asset_map.pb'
      )
      asset_map_path = pathlib.Path(export_path) / 'my_supp_asset_map.pb'
      asset_map = asset_map_pb2.AssetMap.FromString(asset_map_path.read_bytes())
      self.assertLen(asset_map.assets, 1)
      self.assertIn(str(src_asset), asset_map.assets)
      dest_relative_path = asset_map.assets[str(src_asset)]
      self.assertEqual(dest_relative_path, 'my_supp/assets/asset1.txt')
      dest_asset_path = pathlib.Path(export_path) / dest_relative_path
      self.assertTrue(dest_asset_path.exists())
      self.assertEqual(dest_asset_path.read_text(), 'asset_content')

    with self.subTest(name='validate_value'):
      self.assertIn('my_value', manifest_proto.objects)
      val = manifest_proto.objects['my_value'].value.external.data
      self.assertEqual(val.mime_type, 'test/value')
      self.assertEqual(val.inlined_string, 'value_content')

  def test_save_model_with_warmup(self):
    """Tests that save() works with warmup examples."""
    model = jax_model.JaxModel(
        apply_fn_bundle=jax_model.ApplyFnBundle(
            apply_fn=lambda params, inputs: {'out': inputs['x'] * params['w']},
            jax_export_kwargs={'platforms': ['cpu']},
        ),
        params={'w': jnp.array([2.0], dtype=jnp.float32)},
    )
    cfg = serving_config.ServingConfig(
        serving_signature='serving_default',
        input_signature={
            'x': jax.ShapeDtypeStruct(shape=(4,), dtype=jnp.float32)
        },
    )
    export_path = self.create_tempdir().full_path
    warmup_examples = {
        'serving_default': [{'x': jnp.ones((4,), dtype=jnp.float32)}],
    }
    main_lib.save(
        model=model,
        export_path=export_path,
        serving_configs=[cfg],
        warmup_examples=warmup_examples,
        is_grpc_warmup=True,
    )
    warmup_file = (
        pathlib.Path(export_path)
        / 'assets.extra'
        / 'tf_serving_warmup_requests'
    )
    self.assertTrue(warmup_file.exists())


class ValidateSaveParamsTest(absltest.TestCase):

  def test_duplicate_serving_signatures(self):
    with self.assertRaisesRegex(
        ValueError, 'Serving signatures must be unique'
    ):
      main_lib.save(
          jax_model.JaxModel(
              apply_fn_bundle=jax_model.ApplyFnBundle(apply_fn=lambda p, x: x),
          ),
          'test_path',
          [
              serving_config.ServingConfig(
                  serving_signature='serving_default',
                  input_signature={'x': jax.ShapeDtypeStruct((), jnp.float32)},
              ),
              serving_config.ServingConfig(
                  serving_signature='serving_default',
                  input_signature={'x': jax.ShapeDtypeStruct((), jnp.float32)},
              ),
          ],
      )

  def test_missing_input_signature(self):
    with self.assertRaisesRegex(ValueError, 'must have input signature'):
      main_lib.save(
          jax_model.JaxModel(
              apply_fn_bundle=jax_model.ApplyFnBundle(apply_fn=lambda p, x: x),
          ),
          'test_path',
          [
              serving_config.ServingConfig(
                  serving_signature='serving_default',
                  input_signature=None,
              )
          ],
      )

  def test_ambiguous_method_key(self):
    with self.assertRaisesRegex(ValueError, 'does not specify a method key'):
      main_lib.save(
          model=jax_model.JaxModel(
              apply_fn_bundle={
                  'key1': jax_model.ApplyFnBundle(apply_fn=lambda p, x: x),
                  'key2': jax_model.ApplyFnBundle(apply_fn=lambda p, x: x),
              },
          ),
          export_path='test_path',
          serving_configs=[
              serving_config.ServingConfig(
                  serving_signature='serving_default',
                  input_signature={'x': jax.ShapeDtypeStruct((), jnp.float32)},
                  method_key=None,
              )
          ],
      )

  def test_invalid_method_key(self):
    with self.assertRaisesRegex(ValueError, 'does not exist in the JAX model'):
      main_lib.save(
          jax_model.JaxModel(
              apply_fn_bundle=jax_model.ApplyFnBundle(apply_fn=lambda p, x: x),
          ),
          'test_path',
          [
              serving_config.ServingConfig(
                  serving_signature='serving_default',
                  input_signature={'x': jax.ShapeDtypeStruct((), jnp.float32)},
                  method_key='invalid_key',
              )
          ],
      )


class BatchOptionsForProcessorsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'requires_preprocessor',
          orchestration_pb2.BatchOptions.PRE_PROCESSOR_AND_MODEL_FUNCTION,
          (
              'Batch component is set to PRE_PROCESSOR_AND_MODEL_FUNCTION, but'
              ' there are no preprocessors.'
          ),
      ),
      (
          'requires_postprocessor',
          orchestration_pb2.BatchOptions.MODEL_FUNCTION_AND_POST_PROCESSOR,
          (
              'Batch component is set to POST_PROCESSOR_AND_MODEL_FUNCTION, but'
              ' there are no postprocessors.'
          ),
      ),
  )
  def test_batch_component_requires_processors(
      self, batch_component, expected_error
  ):
    with self.assertRaisesRegex(ValueError, expected_error):
      main_lib.save(
          jax_model.JaxModel(
              apply_fn_bundle=jax_model.ApplyFnBundle(apply_fn=lambda p, x: x),
          ),
          'test_path',
          [
              serving_config.ServingConfig(
                  serving_signature='serving_default',
                  input_signature={'x': jax.ShapeDtypeStruct((), jnp.float32)},
                  processing=processor.SequentialProcessing(
                      preprocessors=[], postprocessors=[]
                  ),
                  batch_options=orchestration_pb2.BatchOptions(
                      batch_component=batch_component,
                      max_batch_size=8,
                      max_enqueued_batches=10,
                  ),
              ),
          ],
      )


if __name__ == '__main__':
  absltest.main()
