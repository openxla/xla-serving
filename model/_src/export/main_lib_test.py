"""Tests for neptune.model._src.export.main_lib."""

import pathlib

from absl.testing import absltest
import jax
import jax.numpy as jnp

from google3.net.proto2.contrib.pyutil import compare
from google3.third_party.neptune.model._src.core import persistence_lib
from google3.third_party.neptune.model._src.core import saveable
from google3.third_party.neptune.model._src.core import shlo_type
from google3.third_party.neptune.model._src.export import jax_model
from google3.third_party.neptune.model._src.export import main_lib
from google3.third_party.neptune.model._src.export import serving_config
from google3.third_party.neptune.protos import orchestration_pb2


class SaveModelTest(compare.Proto2Assertions, absltest.TestCase):

  def test_save_model_with_one_config(self):
    """Tests that save() runs."""
    model = jax_model.JaxModel(
        apply_fn=lambda params, x: x * params['w'],
        params={'w': jnp.array([2.0], dtype=jnp.float32)},
    )
    cfg = serving_config.ServingConfig(
        serving_signature='serving_default',
        input_signature=jax.ShapeDtypeStruct(shape=(4,), dtype=jnp.float32),
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
                    elements {
                      dict {
                      }
                    }
                  }
                }
                output {
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
              visibility: PUBLIC
            }
          }
        }
        supplemental_info {
          key: "orchestration"
          value {
            file_system_location {
              string_path: "orchestration"
            }
            mime_type: "application/protobuf; type=neptune_orchestration.Pipelines"
            version: "0.0.1"
          }
        }
      """
      compare.assertProto2Equal(
          self,
          expected_manifest_proto,
          manifest_proto,
          ignored_fields=['objects.function.body.stable_hlo_body'],
      )

    with self.subTest(name='validate_orchestration_proto'):
      expected_pipelines_proto = """
        name_to_pipeline {
          key: "serving_default"
          value {
            components {
              role: ROLE_MODEL
              function_name: "serving_default"
            }
          }
        }
      """
      compare.assertProto2Equal(
          self,
          expected_pipelines_proto,
          pipelines_proto,
      )

  def test_update_save_bundle_for_jax_model_with_polymorphic_shape(self):
    """Tests that _update_save_bundle_for_jax_model works with polymorphic shapes."""
    model = jax_model.JaxModel(
        apply_fn=lambda params, x: x * params['w'],
        params={'w': jnp.array([2.0], dtype=jnp.float32)},
    )
    input_signature = jax.ShapeDtypeStruct(shape=('b', 1), dtype=jnp.float32)
    input_polymophic_shapes = 'b,_'
    serving_signature = 'serving_default'
    save_bundle = main_lib.NeptuneModelSaveBundle()

    input_sig, output_sig = main_lib._update_save_bundle_for_jax_model(
        model=model,
        method_key=None,
        input_signature=input_signature,
        input_polymophic_shapes=input_polymophic_shapes,
        polymorphic_constraints=[],
        serving_signature=serving_signature,
        save_bundle=save_bundle,
    )

    with self.subTest('adds_to_save_bundle'):
      self.assertIn(serving_signature, save_bundle.manifest_objects)
      self.assertIsInstance(
          save_bundle.manifest_objects[serving_signature],
          saveable.Saveable,
      )

    with self.subTest('check_input_signature'):
      # input_sig = ((input_tree,), kwargs_tree)
      # input is ShapeDtypeStruct(shape=('b', 1), dtype=float32)
      self.assertEqual(input_sig[0][0].shape, [None, 1])
      self.assertEqual(input_sig[0][0].dtype, shlo_type.ShloDType.f32)

    with self.subTest('check_output_signature'):
      self.assertEqual(output_sig.shape, [None, 1])
      self.assertEqual(output_sig.dtype, shlo_type.ShloDType.f32)

  def test_calculate_jax_model_input_sig(self):
    params_spec = shlo_type.ShloTensorSpec(
        shape=[1], dtype=shlo_type.ShloDType.f32
    )
    input_spec = shlo_type.ShloTensorSpec(
        shape=[4], dtype=shlo_type.ShloDType.f32
    )
    kw_spec = shlo_type.ShloTensorSpec(shape=[2], dtype=shlo_type.ShloDType.f32)

    with self.subTest(name='with_kwargs'):
      apply_fn_input_sig = ((params_spec, input_spec), {'kw': kw_spec})
      input_sig = main_lib._calculate_jax_model_input_sig(apply_fn_input_sig)
      self.assertEqual(input_sig, ((input_spec,), {'kw': kw_spec}))

    with self.subTest(name='without_kwargs'):
      apply_fn_input_sig = (params_spec, input_spec)
      input_sig = main_lib._calculate_jax_model_input_sig(apply_fn_input_sig)
      self.assertEqual(input_sig, ((input_spec,), {}))

    with self.subTest(name='not_sequence_fails'):
      with self.assertRaisesRegex(ValueError, 'must be a sequence'):
        main_lib._calculate_jax_model_input_sig(params_spec)

    with self.subTest(name='wrong_num_pos_args_fails'):
      with self.assertRaisesRegex(ValueError, 'must have exactly two elements'):
        main_lib._calculate_jax_model_input_sig((params_spec,))


if __name__ == '__main__':
  absltest.main()
