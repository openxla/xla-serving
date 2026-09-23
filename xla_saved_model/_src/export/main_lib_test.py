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

"""Tests for xla_saved_model._src.export.main_lib."""

import pathlib

from absl.testing import absltest
import jax
import jax.numpy as jnp

from xla_saved_model.protos import orchestration_pb2
from xla_saved_model._src.core import persistence_lib
from xla_saved_model._src.core import test_utils
from xla_saved_model._src.export import jax_model
from xla_saved_model._src.export import main_lib
from xla_saved_model._src.export import serving_config
from xla_saved_model._src.export import warmup


class SaveModelTest(test_utils.XsmTestCase):

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
        model,
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
        model,
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
        model,
        export_path=export_path,
        serving_configs=[cfg],
        warmup_examples=warmup_examples,
    )
    warmup_file = (
        pathlib.Path(export_path)
        / warmup.ASSETS_EXTRA_DIR
        / warmup.WARMUP_REQUESTS_FILENAME
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

  def test_multiple_models_without_orchestration(self):
    with self.assertRaisesRegex(
        ValueError, 'When multiple models are provided, orchestration'
    ):
      main_lib.save(
          models=[
              jax_model.JaxModel(
                  apply_fn_bundle=jax_model.ApplyFnBundle(
                      apply_fn=lambda p, x: x
                  ),
                  name='model1',
              ),
              jax_model.JaxModel(
                  apply_fn_bundle=jax_model.ApplyFnBundle(
                      apply_fn=lambda p, x: x
                  ),
                  name='model2',
              ),
          ],
          export_path='test_path',
          serving_configs=[
              serving_config.ServingConfig(
                  serving_signature='serving_default',
                  input_signature={'x': jax.ShapeDtypeStruct((), jnp.float32)},
              )
          ],
      )

  def test_ambiguous_method_key(self):
    with self.assertRaisesRegex(ValueError, 'multiple apply functions'):
      main_lib.save(
          jax_model.JaxModel(
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
              )
          ],
      )

  def test_invalid_method_key(self):
    with self.assertRaisesRegex(ValueError, 'not found in model'):
      main_lib.save(
          jax_model.JaxModel(
              apply_fn_bundle=jax_model.ApplyFnBundle(apply_fn=lambda p, x: x),
          ),
          'test_path',
          [
              serving_config.ServingConfig(
                  serving_signature='serving_default',
                  input_signature={'x': jax.ShapeDtypeStruct((), jnp.float32)},
                  orchestration=serving_config.Orchestration(
                      model_calls=[
                          serving_config.ModelCall(
                              model_name='jax_model', method_key='invalid_key'
                          )
                      ]
                  ),
              )
          ],
      )

  def test_save_model_with_save_shlo_to_file(self):
    """Tests that save() saves StableHLO to a separate file when requested."""
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
    main_lib.save(
        model,
        export_path=export_path,
        serving_configs=[cfg],
        save_shlo_to_file=True,
    )

    manifest_proto = persistence_lib.load(export_path)
    fn = manifest_proto.objects['serving_default'].function
    self.assertTrue(
        fn.body.stable_hlo_body.stable_hlo.HasField('file_system_location')
    )
    shlo_file = (
        fn.body.stable_hlo_body.stable_hlo.file_system_location.string_path
    )
    self.assertTrue(pathlib.Path(export_path, shlo_file).is_file())

  def test_save_model_with_checkpoint_path(self):
    """Tests that save() with checkpoint_path does not save parameters to disk."""
    model = jax_model.JaxModel(
        apply_fn_bundle=jax_model.ApplyFnBundle(
            apply_fn=lambda params, inputs: {'out': inputs['x'] * params['w']},
            jax_export_kwargs={'platforms': ['cpu']},
        ),
        params={'w': jnp.array([2.0], dtype=jnp.float32)},
        checkpoint_path='/cns/path/to/existing/checkpoint',
    )
    cfg = serving_config.ServingConfig(
        serving_signature='serving_default',
        input_signature={
            'x': jax.ShapeDtypeStruct(shape=(4,), dtype=jnp.float32)
        },
    )
    export_path = self.create_tempdir().full_path
    main_lib.save(
        model,
        export_path=export_path,
        serving_configs=[cfg],
    )

    manifest_proto = persistence_lib.load(export_path)
    self.assertIn('weights', manifest_proto.objects)
    weights_val = manifest_proto.objects['weights'].value.external
    self.assertEqual(
        weights_val.data.file_system_location.string_path,
        '/cns/path/to/existing/checkpoint',
    )


if __name__ == '__main__':
  absltest.main()
