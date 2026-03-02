"""Tests for neptune.model._src.export.main_lib."""

from absl.testing import absltest
import jax
import jax.numpy as jnp

from google3.net.proto2.contrib.pyutil import compare
from google3.third_party.neptune.model._src.core import persistence_lib
from google3.third_party.neptune.model._src.export import jax_model
from google3.third_party.neptune.model._src.export import main_lib
from google3.third_party.neptune.model._src.export import serving_config


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
"""
    self.assertProto2Equal(
        expected_manifest_proto,
        manifest_proto,
        ignored_fields=['objects.function.body.stable_hlo_body'],
    )


if __name__ == '__main__':
  absltest.main()
