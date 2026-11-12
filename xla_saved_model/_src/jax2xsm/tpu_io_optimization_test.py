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
from typing import Any, cast

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
    model_inputs_layout: Sequence[Any] | None = None,
    enable_input_r1: bool = True,
    enable_output_r1: bool = True,
) -> tpu_io_optimization.TpuIoOptimizationPlan | None:
  """Builds a `TpuIoOptimizationPlan` directly from `fn`."""
  if num_weight_leaves is None:
    num_weight_leaves = (
        len(jax.tree.leaves(args_spec[0])) if len(args_spec) >= 2 else 0
    )
  return tpu_io_optimization.build_tpu_io_optimization_plan(
      fn,
      args_spec,
      kwargs_spec or {},
      num_weight_leaves=num_weight_leaves,
      model_inputs_layout=model_inputs_layout,
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
    single_dev = jax_sharding.make_single_device_sharding(jax.devices()[0])
    params_spec = {
        "scale": jax.ShapeDtypeStruct(
            (2, 4), jnp.float32, sharding=single_dev
        )
    }
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
    out_eval = jax.eval_shape(
        plan.wrapped_fn, *plan.args_spec, **plan.kwargs_spec
    )
    self.assertEqual(out_eval["target"].shape, expected_out_shape)

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

  def test_unused_user_input_is_skipped(self):
    inputs_spec = {
        "unused": jax.ShapeDtypeStruct((2, 4), jnp.float32),
        "used": jax.ShapeDtypeStruct((2, 4), jnp.float32),
    }
    plan = _build_plan(
        lambda params, inp: inp["used"] * params["w"],
        ({"w": jax.ShapeDtypeStruct((4,), jnp.float32)}, inputs_spec),
        enable_output_r1=False,
    )
    self.assertIsNotNone(plan)
    assert plan is not None
    self.assertEqual(plan.args_spec[1]["unused"].shape, (2, 4))
    self.assertEqual(plan.args_spec[1]["used"].shape, (8,))

  @parameterized.named_parameters(
      (
          "symbolic_shape",
          lambda: (
              lambda params, x: x * params["w"],
              (
                  {"w": jax.ShapeDtypeStruct((4,), jnp.float32)},
                  jax.ShapeDtypeStruct(
                      (jax_export.symbolic_shape("b")[0], 4), jnp.float32
                  ),
              ),
          ),
      ),
      (
          "multi_device_sharding",
          lambda: (
              lambda params, x: x,
              (
                  {},
                  jax.ShapeDtypeStruct(
                      (2, 4),
                      jnp.float32,
                      sharding=jax_sharding.NamedSharding(
                          jax_sharding.Mesh(
                              np.array(jax.devices()[:2]), ("b",)
                          ),
                          jax_sharding.PartitionSpec("b", None),
                      ),
                  ),
              ),
          ),
      ),
      (
          "kwargs_donate_argnames",
          lambda: (
              jax.jit(
                  lambda params, inputs, extra: inputs + extra,
                  donate_argnames="inputs",
              ),
              ({},),
              {
                  "inputs": jax.ShapeDtypeStruct((2, 4), jnp.float32),
                  "extra": jax.ShapeDtypeStruct((2, 4), jnp.float32),
              },
          ),
      ),
      (
          "custom_compiler_options",
          lambda: (
              jax.jit(
                  lambda params, x: x + 1.0,
                  compiler_options={"xla_embed_ir_in_executable": True},
              ),
              ({}, jax.ShapeDtypeStruct((2, 4), jnp.float32)),
          ),
      ),
      (
          "explicit_model_inputs_layout",
          lambda: (
              lambda params, x: x + 1.0,
              ({}, jax.ShapeDtypeStruct((2, 4), jnp.float32)),
              {},
              {"model_inputs_layout": [None, cast(Any, object())]},
          ),
      ),
  )
  def test_unsupported_export_features_return_none(self, build_case):
    case = build_case()
    extra_kwargs = case[3] if len(case) == 4 else {}
    kwargs_spec = case[2] if len(case) >= 3 else None
    self.assertIsNone(
        _build_plan(case[0], case[1], kwargs_spec, **extra_kwargs)
    )


if __name__ == "__main__":
  absltest.main()
