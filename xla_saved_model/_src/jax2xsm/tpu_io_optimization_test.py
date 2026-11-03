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

"""Tests for export-time TPU I/O shape and layout optimizations."""

from collections.abc import Callable, Sequence
import os
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import export as jax_export
from jax import sharding as jax_sharding
import jax.numpy as jnp
import numpy as np

from xla_saved_model._src.jax2xsm import tpu_io_optimization

if "--xla_force_host_platform_device_count" not in os.environ.get(
    "XLA_FLAGS", ""
):
  os.environ["XLA_FLAGS"] = (
      f"{os.environ.get('XLA_FLAGS', '')} "
      "--xla_force_host_platform_device_count=8"
  ).strip()


def _build_plan(
    fn: Callable[..., Any],
    args_spec: Sequence[Any],
    kwargs_spec: dict[str, Any] | None = None,
    *,
    num_weight_leaves: int | None = None,
    enable_input_r1: bool = True,
    enable_output_r1: bool = True,
) -> tpu_io_optimization.TpuIoOptimizationPlan | None:
  """Exports `fn` on CPU and builds a `TpuIoOptimizationPlan`."""
  jitted_fn: Any = fn if hasattr(fn, "trace") else jax.jit(fn)
  kwargs = kwargs_spec or {}
  exported = jax_export.export(jitted_fn, platforms=["cpu"])(
      *args_spec, **kwargs
  )
  if num_weight_leaves is None:
    num_weight_leaves = (
        len(jax.tree.leaves(args_spec[0])) if len(args_spec) >= 2 else 0
    )
  return tpu_io_optimization.build_tpu_io_optimization_plan(
      jitted_fn,
      args_spec,
      kwargs,
      exported,
      num_weight_leaves=num_weight_leaves,
      enable_input_r1_optimization=enable_input_r1,
      enable_output_r1_optimization=enable_output_r1,
  )


class TpuIoOptimizationPlanTest(parameterized.TestCase):

  def test_returns_none_when_all_optimizations_disabled(self):
    plan = _build_plan(
        lambda params, x: x + params["w"],
        (
            {"w": jax.ShapeDtypeStruct((4,), jnp.float32)},
            jax.ShapeDtypeStruct((2, 4), jnp.float32),
        ),
        enable_input_r1=False,
        enable_output_r1=False,
    )
    self.assertIsNone(plan)

  @parameterized.named_parameters(
      ("f32_4d", (2, 3, 5, 2), jnp.float32, False, (60,), (60,)),
      ("weak_f32_2d", (2, 4), jnp.float32, True, (8,), (8,)),
      ("bf16_2d", (3, 4), jnp.bfloat16, False, (12,), (12,)),
      ("bool_3d", (2, 3, 2), jnp.bool_, False, (12,), (12,)),
      ("rank_1_unchanged", (4,), jnp.float32, False, (4,), (4,)),
      ("subbyte_int4_unchanged", (2, 4), jnp.int4, False, (2, 4), (2, 4)),
      ("float0_unchanged", (2, 4), jax.dtypes.float0, False, (2, 4), (2, 4)),
  )
  def test_r1_eligibility_by_rank_and_dtype(
      self,
      shape: tuple[int, ...],
      dtype: Any,
      weak_type: bool,
      expected_in_shape: tuple[int, ...],
      expected_out_shape: tuple[int, ...],
  ):
    params_spec = {"scale": jax.ShapeDtypeStruct((2, 4), jnp.float32)}
    inputs_spec = {
        "anchor": np.zeros((2, 4), dtype=np.float32),
        "target": jax.ShapeDtypeStruct(shape, dtype, weak_type=weak_type),
    }
    plan = _build_plan(
        lambda params, inp: {
            "anchor": inp["anchor"] * params["scale"],
            "target": inp["target"],
        },
        [params_spec, inputs_spec],
    )
    self.assertIsNotNone(plan)
    assert plan is not None
    self.assertEqual(plan.args_spec[0]["scale"].shape, (2, 4))
    self.assertEqual(plan.args_spec[1]["anchor"].shape, (8,))
    self.assertFalse(plan.args_spec[1]["anchor"].weak_type)
    self.assertEqual(plan.args_spec[1]["target"].shape, expected_in_shape)
    self.assertEqual(plan.args_spec[1]["target"].dtype, jnp.dtype(dtype))
    self.assertEqual(plan.args_spec[1]["target"].weak_type, weak_type)
    out_exported = jax_export.export(plan.wrapped_fn, platforms=["cpu"])(
        *plan.args_spec, **plan.kwargs_spec
    )
    out_tree_vals = out_exported.out_tree.unflatten(out_exported.out_avals)
    self.assertEqual(out_tree_vals["target"].shape, expected_out_shape)

  @parameterized.named_parameters(
      ("in_r1_only", True, False),
      ("out_r1_only", False, True),
      ("in_and_out_r1", True, True),
  )
  def test_wrapped_fn_numerical_equivalence(
      self, enable_input_r1: bool, enable_output_r1: bool
  ):
    def apply_fn(params, inputs):
      spatial = jnp.einsum("nhwc,hwo->nhwo", inputs["img"], params["kernel"])
      return {
          "feat": spatial + inputs["bias_2d"][:, None, None, :],
          "summary": jnp.sum(spatial, axis=(1, 2, 3)),
      }

    params = {
        "kernel": jnp.arange(60, dtype=jnp.float32).reshape((3, 5, 4)) * 0.1
    }
    inputs = {
        "img": jnp.arange(60, dtype=jnp.float32).reshape((2, 3, 5, 2)) - 10.0,
        "bias_2d": jnp.arange(8, dtype=jnp.float32).reshape((2, 4)),
    }
    plan = _build_plan(
        apply_fn,
        (params, inputs),
        enable_input_r1=enable_input_r1,
        enable_output_r1=enable_output_r1,
    )
    self.assertIsNotNone(plan)
    assert plan is not None
    self.assertEqual(plan.has_input_optimization, enable_input_r1)
    self.assertEqual(plan.has_output_optimization, enable_output_r1)
    phys_inputs = (
        jax.tree.map(lambda x: x.reshape((-1,)), inputs)
        if enable_input_r1
        else inputs
    )
    raw_out = plan.wrapped_fn(params, phys_inputs)
    expected = apply_fn(params, inputs)
    actual_feat = (
        raw_out["feat"].reshape(expected["feat"].shape)
        if enable_output_r1
        else raw_out["feat"]
    )
    np.testing.assert_allclose(actual_feat, expected["feat"], rtol=1e-5)
    np.testing.assert_allclose(
        raw_out["summary"], expected["summary"], rtol=1e-5
    )

  def test_ineligible_exports_return_none(self):
    s1d = jax.ShapeDtypeStruct((4,), jnp.float32)
    s2d = jax.ShapeDtypeStruct((2, 4), jnp.float32)
    self.assertIsNone(
        _build_plan(lambda params, x: x + params["w"], ({"w": s1d}, s1d))
    )
    sym_2d = jax.ShapeDtypeStruct(
        (jax_export.symbolic_shape("b")[0], 4), jnp.float32
    )
    self.assertIsNone(
        _build_plan(lambda params, x: x * params["w"], ({"w": s1d}, sym_2d))
    )

    mesh = jax_sharding.Mesh(np.array(jax.devices()[:2]), ("b",))
    sharded_2d = jax.ShapeDtypeStruct(
        (2, 4),
        jnp.float32,
        sharding=jax_sharding.NamedSharding(
            mesh, jax_sharding.PartitionSpec("b", None)
        ),
    )
    self.assertIsNone(
        _build_plan(
            jax.jit(
                lambda params, x: x,
                in_shardings=({}, sharded_2d.sharding),
                out_shardings=sharded_2d.sharding,
            ),
            ({}, sharded_2d),
        )
    )

    jitted = jax.jit(lambda x: x + 1.0)
    exported = jax_export.export(jitted, platforms=["cpu"])(s2d)
    self.assertIsNone(
        tpu_io_optimization.build_tpu_io_optimization_plan(
            jitted,
            ({"mismatch": s2d},),
            {},
            exported,
            enable_input_r1_optimization=True,
            enable_output_r1_optimization=True,
        )
    )


if __name__ == "__main__":
  absltest.main()
