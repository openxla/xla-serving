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

import math
import os
from typing import Any
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from google.protobuf import text_format
import jax
from jax.experimental import mesh_utils

from xla_saved_model._src.core import compile_options_util  # pylint: disable=g-bad-import-order


class CompileOptionsUtilTest(parameterized.TestCase):

  @parameterized.product(
      mesh_spec=(
          ((1,), ('x',)),
          ((8,), ('x',)),
          ((4, 2), ('x', 'y')),
          ((2, 2, 2), ('x', 'y', 'z')),
      ),
      platform=('cpu', 'cuda', 'tpu'),
  )
  # This test is prefixed with `test_00_` so that it runs first in
  # the test suite. C++ XLA flag parsing uses `absl::call_once`, meaning
  # XLA flags are permanently cached the first time XLA is initialized.
  # By running this test first and passing `--xla_flags_reset=true`,
  # we ensure clean flag initialization for the mesh creation without
  # being affected by earlier tests.
  def test_00_generate_xla_compile_options_with_jax_mesh(
      self, mesh_spec, platform
  ):
    mesh_shape, mesh_axis_names = mesh_spec
    self.enter_context(
        mock.patch.dict(
            os.environ,
            {
                'XLA_FLAGS': (
                    '--xla_force_host_platform_device_count=8'
                    ' --xla_flags_reset=true'
                )
            },
        )
    )
    devices = jax.devices()[: math.prod(mesh_shape)]
    mesh = jax.sharding.Mesh(
        mesh_utils.create_device_mesh(mesh_shape, devices=devices),
        mesh_axis_names,
    )
    platforms = ['cpu', 'cuda', 'tpu']
    compile_options_map = compile_options_util.generate_xla_compile_options(
        native_serialization_platforms=platforms,
        xla_flags_per_platform={},
        jax_mesh=mesh,
    )
    self.assertLen(compile_options_map.map, len(platforms))
    self.assertIn(platform, compile_options_map.map)
    build_options = compile_options_map.map[platform].executable_build_options
    self.assertEqual(build_options.num_replicas, 1)
    self.assertEqual(build_options.num_partitions, mesh.size)
    self.assertEqual(build_options.device_ordinal, -1)
    self.assertEqual(build_options.process_count, 1)
    self.assertEqual(
        list(build_options.allow_spmd_sharding_propagation_to_parameters),
        [False],
    )
    self.assertEqual(
        list(build_options.allow_spmd_sharding_propagation_to_output),
        [False],
    )
    self.assertTrue(build_options.use_shardy_partitioner)
    device_assignment = build_options.device_assignment
    self.assertEqual(device_assignment.replica_count, 1)
    self.assertEqual(device_assignment.computation_count, mesh.size)
    self.assertLen(device_assignment.computation_devices, mesh.size)
    for i, device in enumerate(device_assignment.computation_devices):
      self.assertEqual(device.replica_device_ids, [i])

  @parameterized.named_parameters(
      dict(testcase_name='cpu', platform='cpu'),
      dict(testcase_name='cuda', platform='cuda'),
      dict(testcase_name='tpu', platform='tpu'),
  )
  def test_generate_xla_compile_options_default_build_options_all_platforms(
      self, platform
  ):
    platforms = ['cpu', 'cuda', 'tpu']
    compile_options_map = compile_options_util.generate_xla_compile_options(
        native_serialization_platforms=platforms,
        persist_xla_flags=False,
    )
    self.assertLen(compile_options_map.map, len(platforms))
    self.assertIn(platform, compile_options_map.map)
    build_options = compile_options_map.map[platform].executable_build_options
    self.assertTrue(build_options.HasField('debug_options'))
    self.assertEqual(build_options.num_replicas, 1)
    self.assertEqual(build_options.num_partitions, 1)
    self.assertEqual(build_options.device_ordinal, -1)
    self.assertEqual(build_options.process_count, 1)
    self.assertEqual(
        list(build_options.allow_spmd_sharding_propagation_to_parameters),
        [False],
    )
    self.assertEqual(
        list(build_options.allow_spmd_sharding_propagation_to_output),
        [False],
    )
    self.assertTrue(build_options.use_shardy_partitioner)
    self.assertFalse(build_options.HasField('device_assignment'))

  def test_generate_xla_compile_options_empty_platforms_returns_empty_map(
      self,
  ):
    compile_options_map = compile_options_util.generate_xla_compile_options(
        native_serialization_platforms=[],
        persist_xla_flags=False,
    )
    self.assertEmpty(compile_options_map.map)

  @parameterized.named_parameters(
      dict(testcase_name='cpu', platform='cpu'),
      dict(testcase_name='cuda', platform='cuda'),
      dict(testcase_name='tpu', platform='tpu'),
  )
  def test_generate_xla_compile_options_mixed_case_and_subset_mesh_devices(
      self, platform
  ):
    # NOTE: Process environment boundary for virtual host CPU device count.
    self.enter_context(
        mock.patch.dict(
            os.environ,
            {
                'XLA_FLAGS': (
                    '--xla_force_host_platform_device_count=8'
                    ' --xla_flags_reset=true'
                )
            },
        )
    )
    all_devices = jax.devices()
    self.assertGreaterEqual(len(all_devices), 2)
    subset_devices = [all_devices[-1], all_devices[0]]
    expected_device_ids = [device.id for device in subset_devices]
    self.assertNotEqual(expected_device_ids[0], expected_device_ids[1])
    mesh = jax.sharding.Mesh(subset_devices, ('tp',))
    compile_options_map = compile_options_util.generate_xla_compile_options(
        native_serialization_platforms=['Cpu', 'CUDA', 'TPU'],
        jax_mesh=mesh,
        persist_xla_flags=False,
    )
    expected_platforms = ['cpu', 'cuda', 'tpu']
    self.assertCountEqual(compile_options_map.map.keys(), expected_platforms)
    platform_options = compile_options_map.map[platform]
    self.assertEmpty(platform_options.env_option_overrides)
    build_options = platform_options.executable_build_options
    self.assertEmpty(build_options.comp_envs.environments)
    self.assertEqual(build_options.num_replicas, 1)
    self.assertEqual(build_options.num_partitions, 2)
    self.assertEqual(
        [
            list(comp_device.replica_device_ids)
            for comp_device in (
                build_options.device_assignment.computation_devices
            )
        ],
        [[device_id] for device_id in expected_device_ids],
    )

  def test_generate_xla_compile_options_snapshots_xla_flags(self):
    self.enter_context(
        mock.patch.dict(
            os.environ,
            {'XLA_FLAGS': '--xla_gpu_autotune_level=4 --xla_flags_reset=true'},
        )
    )
    compile_options_map = compile_options_util.generate_xla_compile_options(
        native_serialization_platforms=['cuda'],
    )
    self.assertIn('cuda', compile_options_map.map)
    compile_options = compile_options_map.map['cuda']
    self.assertEqual(
        compile_options.executable_build_options.debug_options.xla_gpu_autotune_level,
        4,
    )


if __name__ == '__main__':
  absltest.main()
