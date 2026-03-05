"""Tests for neptune.model._src.export.main_lib."""

import pathlib

from absl.testing import absltest
import jax
import jax.numpy as jnp

from google3.net.proto2.contrib.pyutil import compare
from google3.third_party.neptune.model._src.core import persistence_lib
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


if __name__ == '__main__':
  absltest.main()
