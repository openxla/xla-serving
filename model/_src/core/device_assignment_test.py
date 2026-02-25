import os
from absl.testing import absltest
import jax
from jax.experimental import mesh_utils
from jax.experimental import topologies as jax_topologies
import numpy as np
from google3.testing.pybase import parameterized
from google3.third_party.neptune.model._src.core import device_assignment


class DeviceAssignmentTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='id_with_coords_and_core_on_chip',
          assignment=device_assignment.DeviceAssignment(
              id=5, coords=(1, 2), core_on_chip=3
          ),
          expected_id=5,
          expected_coords=(1, 2),
          expected_core_on_chip=3,
      ),
      dict(
          testcase_name='id_only',
          assignment=device_assignment.DeviceAssignment(id=0),
          expected_id=0,
          expected_coords=None,
          expected_core_on_chip=None,
      ),
  )
  def test_mesh_to_assignment_returns_expected(
      self, assignment, expected_id, expected_coords, expected_core_on_chip
  ):
    self.assertEqual(assignment.id, expected_id)
    self.assertEqual(assignment.coords, expected_coords)
    self.assertEqual(assignment.core_on_chip, expected_core_on_chip)

  def test_mesh_to_assignment_for_cpu_returns_id_only(self):
    def _create_mesh() -> jax.sharding.Mesh:
      os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
      devices = mesh_utils.create_device_mesh((2, 2, 2))
      return jax.sharding.Mesh(devices, ('b', 'x', 'y'))

    mesh = _create_mesh()
    result = device_assignment.mesh_to_device_assignment(mesh)

    expected = [
        device_assignment.DeviceAssignment(id=i, coords=None, core_on_chip=None)
        for i in range(8)
    ]
    self.assertEqual(result, expected)

  def test_mesh_to_assignment_for_tpu_returns_expected(self):
    def _create_mesh() -> jax.sharding.Mesh:
      devices = np.array(
          jax_topologies.get_topology_desc('df=4x2').devices
      ).reshape(4, 4)
      return jax.sharding.Mesh(devices, ('x', 'y'))

    mesh = _create_mesh()
    result = device_assignment.mesh_to_device_assignment(mesh)

    expected = [
        device_assignment.DeviceAssignment(
            id=0, coords=[0, 0, 0], core_on_chip=0
        ),
        device_assignment.DeviceAssignment(
            id=1, coords=[0, 0, 0], core_on_chip=1
        ),
        device_assignment.DeviceAssignment(
            id=2, coords=[1, 0, 0], core_on_chip=0
        ),
        device_assignment.DeviceAssignment(
            id=3, coords=[1, 0, 0], core_on_chip=1
        ),
        device_assignment.DeviceAssignment(
            id=8, coords=[0, 1, 0], core_on_chip=0
        ),
        device_assignment.DeviceAssignment(
            id=9, coords=[0, 1, 0], core_on_chip=1
        ),
        device_assignment.DeviceAssignment(
            id=10, coords=[1, 1, 0], core_on_chip=0
        ),
        device_assignment.DeviceAssignment(
            id=11, coords=[1, 1, 0], core_on_chip=1
        ),
        device_assignment.DeviceAssignment(
            id=4, coords=[2, 0, 0], core_on_chip=0
        ),
        device_assignment.DeviceAssignment(
            id=5, coords=[2, 0, 0], core_on_chip=1
        ),
        device_assignment.DeviceAssignment(
            id=6, coords=[3, 0, 0], core_on_chip=0
        ),
        device_assignment.DeviceAssignment(
            id=7, coords=[3, 0, 0], core_on_chip=1
        ),
        device_assignment.DeviceAssignment(
            id=12, coords=[2, 1, 0], core_on_chip=0
        ),
        device_assignment.DeviceAssignment(
            id=13, coords=[2, 1, 0], core_on_chip=1
        ),
        device_assignment.DeviceAssignment(
            id=14, coords=[3, 1, 0], core_on_chip=0
        ),
        device_assignment.DeviceAssignment(
            id=15, coords=[3, 1, 0], core_on_chip=1
        ),
    ]

    self.assertEqual(result, expected)

  def test_mesh_to_assignment_fails_validation(self):
    with self.assertRaisesRegex(
        ValueError,
        'coords and core_on_chip should be either both set or both None',
    ):
      device_assignment.DeviceAssignment(id=0, coords=(1, 2))

    with self.assertRaisesRegex(
        ValueError,
        'coords and core_on_chip should be either both set or both None',
    ):
      device_assignment.DeviceAssignment(id=0, core_on_chip=1)


if __name__ == '__main__':
  absltest.main()
