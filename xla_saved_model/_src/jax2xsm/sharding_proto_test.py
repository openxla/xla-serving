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

import dataclasses
import os
from typing import Any

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

from absl.testing import absltest  # pylint: disable=g-import-not-at-top
from absl.testing import parameterized  # pylint: disable=g-import-not-at-top
from jax.experimental import mesh_utils  # pylint: disable=g-import-not-at-top
from jax.sharding import AbstractDevice  # pylint: disable=g-import-not-at-top
from jax.sharding import AbstractMesh  # pylint: disable=g-import-not-at-top
from jax.sharding import AxisType  # pylint: disable=g-import-not-at-top
from jax.sharding import Mesh  # pylint: disable=g-import-not-at-top
from jax.sharding import NamedSharding  # pylint: disable=g-import-not-at-top
from jax.sharding import PartitionSpec  # pylint: disable=g-import-not-at-top

from xla_saved_model.protos import jax_supplemental_pb2  # pylint: disable=g-import-not-at-top
from xla_saved_model._src.jax2xsm import sharding_proto  # pylint: disable=g-import-not-at-top


@dataclasses.dataclass(frozen=True)
class _LegacyAbstractDevice:
  """Mimics JAX 0.10.0 `AbstractDevice`, which lacked the `platform` field."""

  device_kind: str
  num_cores: int | None


class AbstractMeshProtoTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="empty",
          mesh=AbstractMesh((), ()),
      ),
      dict(
          testcase_name="all_axis_types",
          mesh=AbstractMesh(
              (2, 4, 2),
              ("data", "model", "tp"),
              axis_types=(AxisType.Auto, AxisType.Explicit, AxisType.Manual),
          ),
      ),
      dict(
          testcase_name="with_device_without_num_cores",
          mesh=AbstractMesh(
              (4, 2),
              ("x", "y"),
              axis_types=(AxisType.Auto, AxisType.Auto),
              abstract_device=AbstractDevice(
                  device_kind="cpu", num_cores=None, platform="cpu"
              ),
          ),
      ),
      dict(
          testcase_name="with_device_with_num_cores",
          mesh=AbstractMesh(
              (2, 4),
              ("x", "y"),
              axis_types=(AxisType.Explicit, AxisType.Auto),
              abstract_device=AbstractDevice(
                  device_kind="TPU v5 lite", num_cores=1, platform="tpu"
              ),
          ),
      ),
  )
  def test_round_trip(self, mesh):
    proto = sharding_proto.abstract_mesh_to_proto(mesh)
    restored = sharding_proto.abstract_mesh_from_proto(proto)
    self.assertEqual(restored, mesh)
    self.assertEqual(restored.abstract_device, mesh.abstract_device)

  def test_device_num_cores_presence(self):
    no_cores = sharding_proto.abstract_mesh_to_proto(
        AbstractMesh(
            (2,),
            ("x",),
            abstract_device=AbstractDevice(
                device_kind="cpu", num_cores=None, platform="cpu"
            ),
        )
    )
    self.assertFalse(no_cores.abstract_device.HasField("num_cores"))

    zero_cores = sharding_proto.abstract_mesh_to_proto(
        AbstractMesh(
            (2,),
            ("x",),
            abstract_device=AbstractDevice(
                device_kind="custom", num_cores=0, platform="cpu"
            ),
        )
    )
    self.assertTrue(zero_cores.abstract_device.HasField("num_cores"))
    restored = sharding_proto.abstract_mesh_from_proto(zero_cores)
    self.assertEqual(restored.abstract_device.num_cores, 0)

  def test_legacy_abstract_device_without_platform(self):
    legacy: Any = _LegacyAbstractDevice(device_kind="TPU v4", num_cores=2)
    proto = sharding_proto._abstract_device_to_proto(legacy)
    self.assertEqual(proto.device_kind, "TPU v4")
    self.assertEqual(proto.num_cores, 2)
    self.assertEqual(proto.platform, "")

    restored = sharding_proto._abstract_device_from_proto(
        jax_supplemental_pb2.AbstractDevice(
            device_kind="TPU v4", num_cores=2, platform="tpu"
        ),
        device_cls=_LegacyAbstractDevice,
    )
    self.assertEqual(restored, legacy)

  @parameterized.named_parameters(
      dict(
          testcase_name="missing_axis_name",
          proto=jax_supplemental_pb2.AbstractMesh(
              axis_sizes=[2, 4],
              axis_names=["x"],
              axis_types=[
                  jax_supplemental_pb2.AbstractMesh.AUTO,
                  jax_supplemental_pb2.AbstractMesh.AUTO,
              ],
          ),
          error_regex="same length",
      ),
      dict(
          testcase_name="missing_axis_type",
          proto=jax_supplemental_pb2.AbstractMesh(
              axis_sizes=[2],
              axis_names=["x"],
              axis_types=[],
          ),
          error_regex="same length",
      ),
      dict(
          testcase_name="unspecified_axis_type",
          proto=jax_supplemental_pb2.AbstractMesh(
              axis_sizes=[2],
              axis_names=["x"],
              axis_types=[
                  jax_supplemental_pb2.AbstractMesh.AXIS_TYPE_UNSPECIFIED
              ],
          ),
          error_regex="Unknown AbstractMesh axis type",
      ),
  )
  def test_malformed_mesh_raises(self, proto, error_regex):
    with self.assertRaisesRegex(ValueError, error_regex):
      sharding_proto.abstract_mesh_from_proto(proto)


class NamedShardingsProtoTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name="both_none", in_s=None, out_s=None),
      dict(testcase_name="both_empty", in_s=(), out_s=()),
      dict(
          testcase_name="all_none_leaves",
          in_s=(None, None),
          out_s=(None,),
      ),
  )
  def test_unsharded_produces_no_protos(self, in_s, out_s):
    meshes, in_proto, out_proto = sharding_proto.to_protos(in_s, out_s)
    self.assertEmpty(meshes)
    self.assertIsNone(in_proto)
    self.assertIsNone(out_proto)
    self.assertEqual(sharding_proto.from_proto(in_proto, (), 2), (None, None))

  def test_partition_spec_variants_and_memory_kind_round_trip(self):
    mesh = AbstractMesh(
        (2, 2, 2, 2),
        ("a", "b", "c", "d"),
        axis_types=(
            AxisType.Explicit,
            AxisType.Explicit,
            AxisType.Explicit,
            AxisType.Auto,
        ),
    )
    shardings = (
        NamedSharding(mesh, PartitionSpec()),
        NamedSharding(mesh, PartitionSpec("a", None, ("b", "d"))),
        NamedSharding(
            mesh, PartitionSpec("d", PartitionSpec.UNCONSTRAINED, None)
        ),
        None,
        NamedSharding(
            mesh,
            PartitionSpec("a", reduced=frozenset({"c", "b"})),
        ),
        NamedSharding(
            mesh,
            PartitionSpec("d", unreduced=frozenset({"b", "a"})),
        ),
        NamedSharding(mesh, PartitionSpec("a"), memory_kind="pinned_host"),
        NamedSharding(mesh, PartitionSpec("b"), memory_kind="device"),
    )

    mesh_protos, in_proto, out_proto = sharding_proto.to_protos(shardings, None)

    # All leaves share one mesh, and `reduced` / `unreduced` are sorted for
    # deterministic output.
    self.assertLen(mesh_protos, 1)
    self.assertIsNotNone(in_proto)
    self.assertIsNone(out_proto)
    assert in_proto is not None
    self.assertNotIn(3, in_proto.idx_to_sharding)
    self.assertFalse(in_proto.idx_to_sharding[0].HasField("memory_kind"))
    self.assertEqual(in_proto.idx_to_sharding[6].memory_kind, "pinned_host")
    self.assertEqual(list(in_proto.idx_to_sharding[4].spec.reduced), ["b", "c"])
    self.assertEqual(
        list(in_proto.idx_to_sharding[5].spec.unreduced), ["a", "b"]
    )

    restored_meshes = tuple(
        sharding_proto.abstract_mesh_from_proto(m) for m in mesh_protos
    )
    restored = sharding_proto.from_proto(
        in_proto, restored_meshes, len(shardings)
    )
    self.assertEqual(restored, shardings)

  def test_deduplicates_shared_mesh_and_preserves_multiple_meshes(self):
    concrete_mesh = Mesh(mesh_utils.create_device_mesh((4, 2)), ("x", "y"))
    other_mesh = AbstractMesh((8,), ("data",), axis_types=(AxisType.Explicit,))

    in_shardings = (
        NamedSharding(concrete_mesh, PartitionSpec("x")),
        None,
        NamedSharding(other_mesh, PartitionSpec("data")),
    )
    out_shardings = (
        NamedSharding(concrete_mesh.abstract_mesh, PartitionSpec("x", "y")),
        NamedSharding(other_mesh, PartitionSpec()),
    )

    mesh_protos, in_proto, out_proto = sharding_proto.to_protos(
        in_shardings, out_shardings
    )
    self.assertLen(mesh_protos, 2)
    assert in_proto is not None and out_proto is not None
    self.assertEqual(in_proto.idx_to_sharding[0].abstract_mesh_index, 0)
    self.assertEqual(in_proto.idx_to_sharding[2].abstract_mesh_index, 1)
    self.assertEqual(out_proto.idx_to_sharding[0].abstract_mesh_index, 0)
    self.assertEqual(out_proto.idx_to_sharding[1].abstract_mesh_index, 1)

    restored_meshes = tuple(
        sharding_proto.abstract_mesh_from_proto(m) for m in mesh_protos
    )
    restored_in = sharding_proto.from_proto(in_proto, restored_meshes, 3)
    restored_out = sharding_proto.from_proto(out_proto, restored_meshes, 2)

    expected_in_0 = NamedSharding(
        concrete_mesh.abstract_mesh,
        PartitionSpec("x"),
        memory_kind=in_shardings[0].memory_kind,
    )
    self.assertEqual(restored_in, (expected_in_0, None, in_shardings[2]))
    self.assertEqual(restored_out, out_shardings)

  def test_malformed_shardings_raise(self):
    mesh = AbstractMesh((2,), ("x",), axis_types=(AxisType.Auto,))

    out_of_range_leaf = jax_supplemental_pb2.NamedShardings(
        idx_to_sharding={2: jax_supplemental_pb2.NamedSharding()}
    )
    with self.assertRaisesRegex(ValueError, "leaf index 2 is out of range"):
      sharding_proto.from_proto(out_of_range_leaf, (mesh,), num_leaves=2)

    out_of_range_mesh = jax_supplemental_pb2.NamedShardings(
        idx_to_sharding={
            0: jax_supplemental_pb2.NamedSharding(abstract_mesh_index=1)
        }
    )
    with self.assertRaisesRegex(
        ValueError, "abstract_mesh_index 1 is out of range"
    ):
      sharding_proto.from_proto(out_of_range_mesh, (mesh,), num_leaves=1)

    contradictory_axis = jax_supplemental_pb2.NamedShardings(
        idx_to_sharding={
            0: jax_supplemental_pb2.NamedSharding(
                abstract_mesh_index=0,
                spec=jax_supplemental_pb2.PartitionSpec(
                    partitions=[
                        jax_supplemental_pb2.PartitionSpecAxis(
                            axes=["x"], unconstrained=True
                        )
                    ]
                ),
            )
        }
    )
    with self.assertRaisesRegex(ValueError, "both `unconstrained` and `axes`"):
      sharding_proto.from_proto(contradictory_axis, (mesh,), num_leaves=1)


if __name__ == "__main__":
  absltest.main()
