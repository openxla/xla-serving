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

from absl.testing import absltest
from absl.testing import parameterized
from google.protobuf import text_format
import jax
from jax import export as jax_export
from jax import sharding as jax_sharding
from jax.experimental import mesh_utils
import jax.numpy as jnp
import orbax.checkpoint as ocp

from xla_saved_model.protos import manifest_pb2
from xla_saved_model.protos import type_pb2
from xla_saved_model._src.core import persistence_lib as xsm_core
from xla_saved_model._src.core import test_utils
from xla_saved_model._src.jax2xsm import main_lib
from xla_saved_model._src.test_utils import jax_mnist

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'


def _get_shape_dtype_struct(param):
  """Returns a pytree of `jax.ShapeDtypeStruct`."""

  def _get_leaf_struct(x):
    return jax.ShapeDtypeStruct(x.shape, x.dtype)

  return jax.tree_util.tree_map(_get_leaf_struct, param)


class MainLibTest(test_utils.XsmTestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='_default',
          native_serialization_platforms=None,
          polymorphic_shape=False,
      ),
      dict(
          testcase_name='_polymorphic_shape',
          native_serialization_platforms=None,
          polymorphic_shape=True,
      ),
      dict(
          testcase_name='_native_cpu_serialization',
          native_serialization_platforms=['cpu'],
          polymorphic_shape=False,
      ),
  )
  def test_sharded_jax_e2e(
      self, native_serialization_platforms, polymorphic_shape
  ):

    def _jax_model_fn(params, x):
      return jnp.dot(jnp.dot(x, params['a']), params['b'])

    def _create_mesh() -> jax_sharding.Mesh:
      os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
      devices = mesh_utils.create_device_mesh((4, 2))
      return jax_sharding.Mesh(devices, ('x', 'y'))

    mesh = _create_mesh()
    pytree_shardings = {
        'a': jax_sharding.NamedSharding(mesh, jax_sharding.PartitionSpec('x')),
        'b': jax_sharding.NamedSharding(
            mesh, jax_sharding.PartitionSpec('x', 'y')
        ),
    }

    def _generated_sharded_params():
      a = jnp.array(jax.random.uniform(jax.random.key(0), (4, 4)))
      b = jnp.array(jax.random.uniform(jax.random.key(1), (4, 16)))
      return {
          'a': jax.device_put(a, pytree_shardings['a']),
          'b': jax.device_put(b, pytree_shardings['b']),
      }

    params = _generated_sharded_params()

    in_shardings = (
        pytree_shardings,
        jax_sharding.NamedSharding(
            mesh,
            jax_sharding.PartitionSpec(None, 'x')
            if polymorphic_shape
            else jax_sharding.PartitionSpec('x'),
        ),
    )
    model_fn = jax.jit(_jax_model_fn, in_shardings=in_shardings)

    if polymorphic_shape:
      input_args_spec = jax_export.symbolic_args_specs(
          jax.ShapeDtypeStruct(
              (jax_export.symbolic_shape('b'), 4),
              jnp.float64,
          ),
          shapes_specs='b, 4',
      )
    else:
      input_args_spec = jax.ShapeDtypeStruct((8, 4), jnp.float64)
    params_args_spec = _get_shape_dtype_struct(params)
    jax_export_kwargs = {}
    if native_serialization_platforms:
      jax_export_kwargs['platforms'] = native_serialization_platforms

    obm_shlo_fn = main_lib.convert(
        model_fn,
        (params_args_spec, input_args_spec),
        {},
        jax_export_kwargs=jax_export_kwargs,
    )
    obm_module = dict()
    model_function_name = 'my_model_fn'
    obm_module[model_function_name] = obm_shlo_fn

    save_dir_path = os.path.join(self.create_tempdir())
    ckpt_subdir = 'my_checkpoint/'
    ckpt_path = os.path.join(save_dir_path, ckpt_subdir)
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    checkpointer.save(ckpt_path, params)
    weights_name = 'my_weights'
    obm_module[weights_name] = main_lib.convert_path_to_value(
        ckpt_subdir,
        mime_type='application/x.orbax-checkpoint',
    )

    xsm_core.save(
        obm_module,
        save_dir_path,
        xsm_core.SaveOptions(version=2),
    )

    ckpt_path = os.path.join(save_dir_path, ckpt_subdir)
    self.assertNotEmpty(
        os.listdir(ckpt_path),
        f'Checkpoint directory {ckpt_subdir} is empty.',
    )

    manifest_proto = xsm_core.load(save_dir_path)

    if polymorphic_shape:
      batch_size = ''
      sharding_dims = """
        tile_assignment_dimensions: 1
        tile_assignment_dimensions: 4
        tile_assignment_dimensions: 2
        """
    else:
      sharding_dims = """
        tile_assignment_dimensions: 4
        tile_assignment_dimensions: 1
        tile_assignment_dimensions: 2
      """
      batch_size = 'size: 8'

    expected_signature_text = (
        """
      signature {
        input {
          tuple {
            elements {
              tuple {
                elements {
                  dict {
                    string_to_type {
                      key: "a"
                      value {
                        leaf {
                          tensor_type {
                            shape {
                              shape_with_known_rank {
                                dimension_sizes {
                                  size: 4
                                }
                                dimension_sizes {
                                  size: 4
                                }
                              }
                            }
                            dtype: f32
                            sharding {
                              type: OTHER
                              tile_assignment_dimensions: 4
                              tile_assignment_dimensions: 1
                              tile_assignment_dimensions: 2
                              replicate_on_last_tile_dim: true
                              iota_reshape_dims: 8
                              iota_transpose_perm: 0
                            }
                          }
                        }
                      }
                    }
                    string_to_type {
                      key: "b"
                      value {
                        leaf {
                          tensor_type {
                            shape {
                              shape_with_known_rank {
                                dimension_sizes {
                                  size: 4
                                }
                                dimension_sizes {
                                  size: 16
                                }
                              }
                            }
                            dtype: f32
                            sharding {
                              type: OTHER
                              tile_assignment_dimensions: 4
                              tile_assignment_dimensions: 2
                              iota_reshape_dims: 8
                              iota_transpose_perm: 0
                            }
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
                            """
        + batch_size
        + """
                          }
                          dimension_sizes {
                            size: 4
                          }
                        }
                      }
                      dtype: f32
                      sharding {
                        type: OTHER
                        """
        + sharding_dims
        + """
                        replicate_on_last_tile_dim: true
                        iota_reshape_dims: 8
                        iota_transpose_perm: 0
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
          leaf {
            tensor_type {
              shape {
                shape_with_known_rank {
                  dimension_sizes {
                    """
        + batch_size
        + """
                  }
                  dimension_sizes {
                    size: 16
                  }
                }
              }
              dtype: f32
            }
          }
        }
      }
    """
    )

    expected_manifest_proto_text = (
        """
      objects {
        key: \""""
        + model_function_name
        + """\"
        value {
          function {"""
        + expected_signature_text
        + """
            body {
              stable_hlo_body {
                stable_hlo {
                  inlined_bytes: "ML StableHLO v0.9.0 ..."
                  mime_type: "application/x.mlir-stablehlo"
                  version: "1.0"
                }
                calling_convention_version: 10
                lowering_platforms: \""""
        + ('cpu' if native_serialization_platforms else 'tpu')
        + """\"
                module_kept_var_idx: 0
                module_kept_var_idx: 1
                module_kept_var_idx: 2
                supplemental_info {
                  key: "jax_specific_info"
                  value {
                    file_system_location {
                      string_path: \""""
        + model_function_name
        + """_jax_specific_info_supplemental.pb\"
                    }
                    mime_type: "application/protobuf; type=orbax_model_jax_supplemental.Function"
                    version: "0.0.1"
                  }
                }
              }
            }
            visibility: PUBLIC
          }
        }
      }
      objects {
        key: \""""
        + weights_name
        + """\"
        value {
          value {
            external {
              data {
                file_system_location {
                  string_path: \""""
        + ckpt_subdir
        + """\"
                }
                mime_type: "application/x.orbax-checkpoint"
              }
            }
          }
        }
      }
      """
    )
    expected_manifest_proto = text_format.Parse(
        expected_manifest_proto_text, manifest_pb2.Manifest()
    )

    self.assertProtosEqual(
        manifest_proto,
        expected_manifest_proto,
        ignored_fields=[
            'objects.function.body.stable_hlo_body.stable_hlo.inlined_bytes'
        ],
    )

    self.assertIn(
        b'ML\xef',
        manifest_proto.objects[
            model_function_name
        ].function.body.stable_hlo_body.stable_hlo.inlined_bytes,
    )
    self.assertIn(
        b'StableHLO_v',
        manifest_proto.objects[
            model_function_name
        ].function.body.stable_hlo_body.stable_hlo.inlined_bytes,
    )

  def test_mnist_model(self):

    model = jax_mnist.JaxMnist()
    input_args_spec = jax.ShapeDtypeStruct((4, 28, 28, 1), jnp.float64)
    params = model.init(
        jax.random.PRNGKey(123),
        jnp.ones(shape=input_args_spec.shape, dtype=input_args_spec.dtype),
    )

    def get_mesh():
      devices = mesh_utils.create_device_mesh((2, 2, 2))
      return jax_sharding.Mesh(devices, ('b', 'x', 'y'))

    mesh = get_mesh()
    params_sharding_spec = jax.tree_util.tree_map(
        lambda _: jax_sharding.NamedSharding(
            mesh, jax_sharding.PartitionSpec('y')
        ),
        params,
    )
    in_shardings_ = (
        params_sharding_spec,
        jax_sharding.NamedSharding(
            mesh, jax_sharding.PartitionSpec('b', 'x', None, None)
        ),
    )

    model_apply_fn = jax.jit(
        model.apply,
        in_shardings=in_shardings_,
        out_shardings=jax_sharding.NamedSharding(
            mesh, jax_sharding.PartitionSpec('b', 'y')
        ),
    )

    params_args_spec = _get_shape_dtype_struct(params)

    em_shlo_fn = main_lib.convert(
        model_apply_fn,
        (params_args_spec, input_args_spec),
        {},
    )

    obm_module = dict()

    model_function_name = 'mnist_forward_fn'
    obm_module[model_function_name] = em_shlo_fn
    save_dir_path = os.path.join(self.create_tempdir())

    checkpoint_path = 'my_checkpoint/'
    checkpoint_abs_path = os.path.join(save_dir_path, checkpoint_path)
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    checkpointer.save(checkpoint_abs_path, params)
    obm_module['my_weights'] = main_lib.convert_path_to_value(
        checkpoint_path,
        mime_type='application/x.orbax-checkpoint',
    )

    xsm_core.save(
        obm_module,
        save_dir_path,
        xsm_core.SaveOptions(
            version=2,
        ),
    )

    manifest_proto = xsm_core.load(save_dir_path)

    manifest_path = (
        pathlib.Path(__file__).parent.parent
        / 'test_utils'
        / 'mnist_jax_model_expected_manifest.textproto'
    )
    expected_manifest_proto_text = manifest_path.read_text().replace(
        'MODEL_FN_NAME_PLACEHOLDER', model_function_name
    )
    expected_manifest_proto = text_format.Parse(
        expected_manifest_proto_text, manifest_pb2.Manifest()
    )

    self.assertProtosEqual(
        manifest_proto,
        expected_manifest_proto,
        ignored_fields=[
            'objects.function.body.stable_hlo_body.stable_hlo.inlined_bytes'
        ],
    )
    self.assertNotEmpty(
        os.listdir(checkpoint_abs_path),
        f"Checkpoint directory '{checkpoint_abs_path}' is empty.",
    )

  def test_host_placement_e2e(self):
    def _jax_model_fn(params, x):
      return jnp.dot(x, params['w'])

    devices = mesh_utils.create_device_mesh((4, 2))
    mesh = jax_sharding.Mesh(devices, ('x', 'y'))
    host_sharding = jax_sharding.NamedSharding(
        mesh, jax_sharding.PartitionSpec('x'), memory_kind='pinned_host'
    )
    device_sharding = jax_sharding.NamedSharding(
        mesh, jax_sharding.PartitionSpec('x')
    )

    params = {'w': jnp.zeros((4, 4), dtype=jnp.float32)}
    in_shardings = ({'w': host_sharding}, device_sharding)
    model_fn = jax.jit(_jax_model_fn, in_shardings=in_shardings)

    input_args_spec = jax.ShapeDtypeStruct((4, 4), jnp.float32)
    params_args_spec = _get_shape_dtype_struct(params)

    obm_shlo_fn = main_lib.convert(
        model_fn,
        (params_args_spec, input_args_spec),
        {},
        jax_export_kwargs={'platforms': ['cpu']},
    )

    shlo_params_spec = obm_shlo_fn.input_signature[0][0]['w']
    self.assertEqual(shlo_params_spec.placement, type_pb2.HOST)

    save_dir_path = self.create_tempdir().full_path
    obm_module = {'model_fn': obm_shlo_fn}
    xsm_core.save(
        obm_module,
        save_dir_path,
        xsm_core.SaveOptions(version=2),
    )

    manifest_proto = xsm_core.load(save_dir_path)
    fn_sig = manifest_proto.objects['model_fn'].function.signature
    input_tuple = fn_sig.input.tuple.elements[0].tuple.elements[0]
    w_type = input_tuple.dict.string_to_type['w'].leaf.tensor_type
    self.assertEqual(w_type.placement, type_pb2.HOST)


if __name__ == '__main__':
  absltest.main()
