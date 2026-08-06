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
from typing import Any
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from google.protobuf import text_format
import jax
from jax.experimental import mesh_utils
from neptune_model._src.core import compile_options_util
from neptune_model.protos.xla.hlo.transforms.simplifiers import memory_scheduler_pb2 as memory_scheduler_proto_pb2  # pylint: disable=g-direct-tensorflow-import


class CompileOptionsUtilTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='1d_mesh',
          mesh_shape=(8,),
          mesh_axis_names=('x',),
      ),
      dict(
          testcase_name='2d_mesh',
          mesh_shape=(4, 2),
          mesh_axis_names=('x', 'y'),
      ),
      dict(
          testcase_name='3d_mesh',
          mesh_shape=(2, 2, 2),
          mesh_axis_names=('x', 'y', 'z'),
      ),
  )
  # This test is prefixed with `test_00_` so that it runs first in
  # the test suite. C++ XLA flag parsing uses `absl::call_once`, meaning
  # XLA flags are permanently cached the first time XLA is initialized.
  # By running this test first and passing `--xla_flags_reset=true`,
  # we ensure clean flag initialization for the mesh creation without
  # being affected by earlier tests.
  def test_00_generate_xla_compile_options_with_jax_mesh(
      self, mesh_shape, mesh_axis_names
  ):
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
    mesh = jax.sharding.Mesh(
        mesh_utils.create_device_mesh(mesh_shape), mesh_axis_names
    )
    compile_options_map = compile_options_util.generate_xla_compile_options(
        native_serialization_platforms=['tpu'],
        xla_flags_per_platform={},
        jax_mesh=mesh,
    )
    self.assertIn('tpu', compile_options_map.map)
    compile_options = compile_options_map.map['tpu']
    self.assertEqual(
        compile_options.executable_build_options.num_partitions, mesh.size
    )
    device_assignment = (
        compile_options.executable_build_options.device_assignment
    )
    self.assertEqual(device_assignment.replica_count, 1)
    self.assertEqual(device_assignment.computation_count, mesh.size)
    self.assertLen(device_assignment.computation_devices, mesh.size)
    for i, device in enumerate(device_assignment.computation_devices):
      self.assertEqual(device.replica_device_ids, [i])

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
