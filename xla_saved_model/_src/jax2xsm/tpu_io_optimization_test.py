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

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import export as jax_export
from jax import sharding as jax_sharding
from jax.experimental import layout as jax_layout
from jax.experimental import mesh_utils
import jax.extend as jex
import jax.numpy as jnp
import numpy as np

from xla_saved_model.protos import manifest_pb2
from xla_saved_model.protos import type_pb2
from xla_saved_model._src.core import shlo_type
from xla_saved_model._src.jax2xsm import main_lib
from xla_saved_model._src.jax2xsm import tpu_io_optimization
from xla_saved_model._src.jax2xsm import typing


def _simulate_runtime_call(
    shlo_fn,
    params,
    inputs,
):
  """Executes `shlo_fn` on CPU simulating C++ serving runtime I/O adapters."""
  in_specs, in_tree = jax.tree.flatten(
      shlo_fn.optimized_input_signature or shlo_fn.input_signature
  )
  out_specs, out_tree = jax.tree.flatten(
      shlo_fn.optimized_output_signature or shlo_fn.output_signature
  )
  logical_out_specs, _ = jax.tree.flatten(shlo_fn.output_signature)

  def _spec_to_hlo_sharding(spec):
    if spec.sharding is None:
      return None
    return jex.sharding.get_hlo_sharding_from_serialized_proto(
        spec.sharding.SerializeToString()
    )

  exported = jax_export.Exported(
      fun_name="test_fn",
      in_tree=in_tree,
      in_avals=tuple(
          jax.core.ShapedArray(
              tuple(spec.shape),
              shlo_type.shlo_dtype_to_np_dtype(spec.dtype),
          )
          for spec in in_specs
      ),
      out_tree=out_tree,
      out_avals=tuple(
          jax.core.ShapedArray(
              tuple(spec.shape),
              shlo_type.shlo_dtype_to_np_dtype(spec.dtype),
          )
          for spec in out_specs
      ),
      in_shardings_hlo=tuple(_spec_to_hlo_sharding(s) for s in in_specs),
      out_shardings_hlo=tuple(_spec_to_hlo_sharding(s) for s in out_specs),
      nr_devices=1,
      ordered_effects=(),
      unordered_effects=(),
      disabled_safety_checks=(jax_export.DisabledSafetyCheck.platform(),),
      mlir_module_serialized=shlo_fn.mlir_module_serialized,
      calling_convention_version=shlo_fn.calling_convention_version,
      module_kept_var_idx=tuple(shlo_fn.module_kept_var_idx),
      platforms=tuple(shlo_fn.lowering_platforms),
      uses_global_constants=False,
      _get_vjp=None,
      _in_named_shardings=(None,) * len(in_specs),
      _out_named_shardings=(None,) * len(out_specs),
  )

  call_leaves, _ = jax.tree.flatten(((params, inputs), {}))
  adapted_leaves = []
  for leaf, spec in zip(call_leaves, in_specs, strict=True):
    if spec.host_transpose:
      leaf = jnp.transpose(leaf, spec.host_transpose)
    if jnp.shape(leaf) != tuple(spec.shape):
      leaf = jnp.reshape(leaf, tuple(spec.shape))
    adapted_leaves.append(leaf)

  (call_params, call_inputs), _ = in_tree.unflatten(adapted_leaves)
  phys_out = exported.call(call_params, call_inputs)
  phys_out_leaves = out_tree.flatten_up_to(phys_out)
  logical_out_leaves = [
      jnp.reshape(leaf, tuple(logical_spec.shape))
      if jnp.shape(leaf) != tuple(logical_spec.shape)
      else leaf
      for leaf, logical_spec in zip(
          phys_out_leaves, logical_out_specs, strict=True
      )
  ]
  return out_tree.unflatten(logical_out_leaves)


class TpuIoOptimizationTest(parameterized.TestCase):

  def test_input_r1_flattens_user_inputs_and_skips_weights(self):
    def apply_fn(params, inputs):
      return jnp.dot(inputs["x"], params["w"]) + inputs["bias_1d"]

    params_spec = {"w": jax.ShapeDtypeStruct((4, 8), jnp.float32)}
    inputs_spec = {
        "x": jax.ShapeDtypeStruct((2, 4), jnp.float32),
        "bias_1d": jax.ShapeDtypeStruct((8,), jnp.float32),
    }

    shlo_fn = main_lib.convert(
        apply_fn,
        (params_spec, inputs_spec),
        {},
        enable_input_r1_optimization=True,
    )

    self.assertEqual(shlo_fn.input_signature[0][0]["w"].shape, (4, 8))
    self.assertEqual(shlo_fn.input_signature[0][1]["x"].shape, (2, 4))
    self.assertEqual(shlo_fn.input_signature[0][1]["bias_1d"].shape, (8,))
    opt_in_sig = shlo_fn.optimized_input_signature
    self.assertIsNotNone(opt_in_sig)
    self.assertEqual(opt_in_sig[0][0]["w"].shape, (4, 8))
    self.assertEqual(opt_in_sig[0][1]["x"].shape, (8,))
    self.assertEqual(opt_in_sig[0][1]["bias_1d"].shape, (8,))
    self.assertIsNone(shlo_fn.optimized_output_signature)
    self.assertIn(b"reshape_v1", shlo_fn.mlir_module_serialized)

  def test_output_r1_flattens_rank_ge_2_outputs(self):
    def apply_fn(params, x):
      return {"out_2d": jnp.dot(x, params["w"]), "out_1d": jnp.sum(x, axis=1)}

    params_spec = {"w": jax.ShapeDtypeStruct((4, 6), jnp.float32)}
    input_spec = jax.ShapeDtypeStruct((2, 4), jnp.float32)

    shlo_fn = main_lib.convert(
        apply_fn,
        (params_spec, input_spec),
        {},
        enable_output_r1_optimization=True,
    )

    self.assertEqual(shlo_fn.output_signature["out_2d"].shape, (2, 6))
    self.assertEqual(shlo_fn.output_signature["out_1d"].shape, (2,))
    self.assertIsNone(shlo_fn.optimized_input_signature)
    opt_out_sig = shlo_fn.optimized_output_signature
    self.assertIsNotNone(opt_out_sig)
    self.assertEqual(opt_out_sig["out_2d"].shape, (12,))
    self.assertEqual(opt_out_sig["out_1d"].shape, (2,))

  def test_nchw_transpose_and_input_r1_composition(self):
    params_spec = {"scale": jax.ShapeDtypeStruct((2, 3, 5, 2), jnp.float32)}
    inputs_spec = {
        # Asymmetric (N=2, H=3, W=5, C=2): C <= 3 and C < W, eligible.
        "img": jax.ShapeDtypeStruct((2, 3, 5, 2), jnp.float32),
        # Ineligible for NCHW (C=5 >= W=5), but eligible for R1.
        "wide_c": jax.ShapeDtypeStruct((2, 3, 5, 5), jnp.float32),
        # Ineligible for NCHW (bool dtype), but eligible for R1.
        "mask": jax.ShapeDtypeStruct((2, 3, 5, 2), jnp.bool_),
    }

    def apply_fn_mixed(params, inputs):
      return (
          inputs["img"] * params["scale"]
          + inputs["wide_c"][..., :2]
          + inputs["mask"].astype(jnp.float32)
      )

    shlo_fn_nchw_only = main_lib.convert(
        apply_fn_mixed,
        (params_spec, inputs_spec),
        {},
        enable_input_r1_optimization=False,
        max_transpose_channels=3,
    )
    self.assertEqual(
        shlo_fn_nchw_only.input_signature[0][1]["img"].shape, (2, 3, 5, 2)
    )
    self.assertIsNone(
        shlo_fn_nchw_only.input_signature[0][1]["img"].host_transpose
    )
    opt_in_nchw = shlo_fn_nchw_only.optimized_input_signature
    self.assertIsNotNone(opt_in_nchw)
    self.assertEqual(opt_in_nchw[0][1]["img"].shape, (2, 2, 3, 5))
    self.assertEqual(
        opt_in_nchw[0][1]["img"].host_transpose,
        tpu_io_optimization.NHWC_TO_NCHW_PERM,
    )
    self.assertEqual(opt_in_nchw[0][1]["wide_c"].shape, (2, 3, 5, 5))
    self.assertIsNone(opt_in_nchw[0][1]["wide_c"].host_transpose)
    self.assertIsNone(opt_in_nchw[0][1]["mask"].host_transpose)

    shlo_fn_both = main_lib.convert(
        apply_fn_mixed,
        (params_spec, inputs_spec),
        {},
        enable_input_r1_optimization=True,
        max_transpose_channels=3,
    )
    self.assertEqual(
        shlo_fn_both.input_signature[0][1]["img"].shape, (2, 3, 5, 2)
    )
    self.assertIsNone(shlo_fn_both.input_signature[0][1]["img"].host_transpose)
    opt_in_both = shlo_fn_both.optimized_input_signature
    self.assertIsNotNone(opt_in_both)
    self.assertEqual(opt_in_both[0][1]["img"].shape, (60,))
    self.assertEqual(
        opt_in_both[0][1]["img"].host_transpose,
        tpu_io_optimization.NHWC_TO_NCHW_PERM,
    )
    self.assertEqual(opt_in_both[0][1]["wide_c"].shape, (150,))
    self.assertIsNone(opt_in_both[0][1]["wide_c"].host_transpose)

  @parameterized.named_parameters(
      ("in_r1_only", True, False, 0),
      ("out_r1_only", False, True, 0),
      ("nchw_only", False, False, 3),
      ("nchw_and_in_r1", True, False, 3),
      ("all_optimizations", True, True, 3),
  )
  def test_numerical_equivalence_on_asymmetric_nhwc_tensors(
      self,
      enable_input_r1: bool,
      enable_output_r1: bool,
      max_channels: int,
  ):
    def apply_fn(params, inputs):
      spatial = jnp.einsum("nhwc,hwo->nhwo", inputs["img"], params["kernel"])
      return {
          "feat": spatial + inputs["bias_2d"][:, None, None, :],
          "summary": jnp.sum(spatial, axis=(1, 2, 3)),
      }

    params = {
        "kernel": (
            jnp.arange(3 * 5 * 4, dtype=jnp.float32).reshape((3, 5, 4)) * 0.1
        ),
    }
    inputs = {
        "img": (
            jnp.arange(2 * 3 * 5 * 2, dtype=jnp.float32).reshape((2, 3, 5, 2))
            - 10.0
        ),
        "bias_2d": jnp.arange(2 * 4, dtype=jnp.float32).reshape((2, 4)),
    }
    params_spec = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), params
    )
    inputs_spec = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), inputs
    )

    shlo_fn = main_lib.convert(
        jax.jit(apply_fn),
        (params_spec, inputs_spec),
        {},
        platforms=[manifest_pb2.Platform.CPU],
        model_param_names=["kernel"],
        enable_input_r1_optimization=enable_input_r1,
        enable_output_r1_optimization=enable_output_r1,
        max_transpose_channels=max_channels,
    )

    expected = apply_fn(params, inputs)
    actual = _simulate_runtime_call(shlo_fn, params, inputs)
    np.testing.assert_allclose(actual["feat"], expected["feat"], rtol=1e-5)
    np.testing.assert_allclose(
        actual["summary"], expected["summary"], rtol=1e-5
    )

  def test_sharding_collapses_batch_and_executes_identically(self):
    devices = mesh_utils.create_device_mesh((4, 2))
    mesh = jax_sharding.Mesh(devices, ("b", "m"))

    batch_sharded = jax_sharding.NamedSharding(
        mesh, jax_sharding.PartitionSpec("b", None, None, None)
    )
    non_batch_sharded = jax_sharding.NamedSharding(
        mesh, jax_sharding.PartitionSpec("b", None, None, "m")
    )
    weight_sharded = jax_sharding.NamedSharding(
        mesh, jax_sharding.PartitionSpec(None, "m")
    )

    def apply_fn(params, inputs):
      proj = jnp.einsum("nhwc,cm->nhwm", inputs["batch_only"], params["w"])
      return proj + inputs["non_batch"]

    jitted_fn = jax.jit(
        apply_fn,
        in_shardings=(
            {"w": weight_sharded},
            {"batch_only": batch_sharded, "non_batch": non_batch_sharded},
        ),
        out_shardings=batch_sharded,
    )

    params = {
        "w": jax.device_put(
            jnp.arange(2 * 4, dtype=jnp.float32).reshape((2, 4)),
            weight_sharded,
        )
    }
    inputs = {
        "batch_only": jax.device_put(
            jnp.arange(4 * 3 * 5 * 2, dtype=jnp.float32).reshape((4, 3, 5, 2)),
            batch_sharded,
        ),
        "non_batch": jax.device_put(
            jnp.ones((4, 3, 5, 4), dtype=jnp.float32), non_batch_sharded
        ),
    }
    params_spec = {"w": jax.ShapeDtypeStruct((2, 4), jnp.float32)}
    inputs_spec = {
        "batch_only": jax.ShapeDtypeStruct((4, 3, 5, 2), jnp.float32),
        "non_batch": jax.ShapeDtypeStruct((4, 3, 5, 4), jnp.float32),
    }

    shlo_fn = main_lib.convert(
        jitted_fn,
        (params_spec, inputs_spec),
        {},
        platforms=[manifest_pb2.Platform.CPU],
        model_param_names=["w"],
        enable_input_r1_optimization=True,
        enable_output_r1_optimization=True,
        max_transpose_channels=3,
    )

    opt_in_sig = shlo_fn.optimized_input_signature
    self.assertIsNotNone(opt_in_sig)
    self.assertEqual(opt_in_sig[0][1]["batch_only"].shape, (120,))
    self.assertEqual(
        opt_in_sig[0][1]["batch_only"].host_transpose, (0, 3, 1, 2)
    )
    self.assertEqual(
        list(
            opt_in_sig[0][1]["batch_only"].sharding.tile_assignment_dimensions
        ),
        [4, 2],
    )
    # `non_batch` has non-batch sharding on axis 3 -> skipped cleanly.
    self.assertEqual(opt_in_sig[0][1]["non_batch"].shape, (4, 3, 5, 4))
    self.assertIsNone(opt_in_sig[0][1]["non_batch"].host_transpose)
    opt_out_sig = shlo_fn.optimized_output_signature
    self.assertIsNotNone(opt_out_sig)
    self.assertEqual(opt_out_sig.shape, (240,))

    expected = jitted_fn(params, inputs)
    actual = _simulate_runtime_call(shlo_fn, params, inputs)
    np.testing.assert_allclose(actual, expected, rtol=1e-5)

  def test_single_device_sharding_and_concrete_array_specs_succeed(self):
    cpu_device = jax.devices("cpu")[0]
    single_sharding = jax_sharding.SingleDeviceSharding(cpu_device)
    concrete_w = jax.device_put(
        jnp.ones((2, 4), dtype=jnp.float32), single_sharding
    )
    concrete_x = jax.device_put(
        jnp.ones((2, 3, 5, 2), dtype=jnp.float32), single_sharding
    )

    jitted_fn = jax.jit(
        lambda p, x: jnp.einsum("nhwc,co->nhwo", x, p["w"]),
        backend="cpu",
    )
    shlo_fn = main_lib.convert(
        jitted_fn,
        ({"w": concrete_w}, concrete_x),
        {},
        platforms=[manifest_pb2.Platform.CPU],
        model_param_names=["w"],
        enable_input_r1_optimization=True,
        enable_output_r1_optimization=True,
        max_transpose_channels=3,
    )

    opt_in_sig = shlo_fn.optimized_input_signature
    opt_out_sig = shlo_fn.optimized_output_signature
    self.assertIsNotNone(opt_in_sig)
    self.assertIsNotNone(opt_out_sig)
    self.assertEqual(opt_in_sig[0][1].shape, (60,))
    self.assertEqual(opt_in_sig[0][1].host_transpose, (0, 3, 1, 2))
    self.assertEqual(opt_out_sig.shape, (120,))

  def test_custom_format_layout_skips_leaf_and_preserves_weight_layout(self):
    devices = mesh_utils.create_device_mesh((2,), devices=jax.devices()[:2])
    mesh = jax_sharding.Mesh(devices, ("b",))
    replicated = jax_sharding.NamedSharding(mesh, jax_sharding.PartitionSpec())
    custom_layout = jax_layout.Layout(major_to_minor=(0, 1))
    custom_format = jax_layout.Format(custom_layout, replicated)

    jitted_fn = jax.jit(
        lambda p, inputs: jnp.dot(inputs["custom"], p["w"]) + inputs["plain"],
        in_shardings=(
            {"w": custom_format},
            {"custom": custom_format, "plain": replicated},
        ),
        out_shardings=custom_format,
    )
    params_spec = {"w": jax.ShapeDtypeStruct((4, 6), jnp.float32)}
    inputs_spec = {
        "custom": jax.ShapeDtypeStruct((2, 4), jnp.float32),
        "plain": jax.ShapeDtypeStruct((2, 6), jnp.float32),
    }

    shlo_fn = main_lib.convert(
        jitted_fn,
        (params_spec, inputs_spec),
        {},
        model_param_names=["w"],
        model_inputs_layout=[custom_layout],
        enable_input_r1_optimization=True,
        enable_output_r1_optimization=True,
    )

    # `custom` input has explicit Format -> skipped; `plain` input -> flattened;
    # output has explicit Format -> skipped; prefix `model_inputs_layout` works.
    opt_in_sig = shlo_fn.optimized_input_signature
    self.assertIsNotNone(opt_in_sig)
    self.assertEqual(opt_in_sig[0][1]["custom"].shape, (2, 4))
    self.assertEqual(opt_in_sig[0][1]["plain"].shape, (12,))
    self.assertIsNone(shlo_fn.optimized_output_signature)
    self.assertEqual(
        list(shlo_fn.input_signature[0][0]["w"].layout.minor_to_major), [1, 0]
    )

  def test_compiler_options_and_keep_unused_are_preserved(self):
    def apply_fn(params, inputs):
      del inputs["unused"]
      return inputs["used"] * params["w"]

    jitted_fn = jax.jit(
        apply_fn,
        keep_unused=True,
        compiler_options={"xla_embed_ir_in_executable": True},
    )
    params_spec = {"w": jax.ShapeDtypeStruct((1, 1, 1, 2), jnp.float32)}
    inputs_spec = {
        "unused": jax.ShapeDtypeStruct((2, 3, 5, 2), jnp.float32),
        "used": jax.ShapeDtypeStruct((2, 3, 5, 2), jnp.float32),
    }

    shlo_fn = main_lib.convert(
        jitted_fn,
        (params_spec, inputs_spec),
        {},
        model_param_names=["w"],
        enable_input_r1_optimization=True,
        enable_output_r1_optimization=True,
        max_transpose_channels=3,
    )

    self.assertEqual(list(shlo_fn.module_kept_var_idx), [0, 1, 2])
    opt_in_sig = shlo_fn.optimized_input_signature
    self.assertIsNotNone(opt_in_sig)
    self.assertEqual(opt_in_sig[0][1]["used"].shape, (60,))
    self.assertEqual(opt_in_sig[0][1]["unused"].shape, (60,))

  def test_pruned_unused_input_is_not_marked_optimized(self):
    def apply_fn(params, inputs):
      return inputs["used"] * params["w"]

    params_spec = {"w": jax.ShapeDtypeStruct((1, 1, 1, 2), jnp.float32)}
    inputs_spec = {
        "unused": jax.ShapeDtypeStruct((2, 3, 5, 2), jnp.float32),
        "used": jax.ShapeDtypeStruct((2, 3, 5, 2), jnp.float32),
    }

    shlo_fn = main_lib.convert(
        jax.jit(apply_fn, keep_unused=False),
        (params_spec, inputs_spec),
        {},
        model_param_names=["w"],
        enable_input_r1_optimization=True,
        max_transpose_channels=3,
    )

    opt_in = shlo_fn.optimized_input_signature
    self.assertIsNotNone(opt_in)
    # `unused` was pruned by dead-arg elimination -> kept logical; `used` is R1.
    self.assertEqual(opt_in[0][1]["unused"].shape, (2, 3, 5, 2))
    self.assertIsNone(opt_in[0][1]["unused"].host_transpose)
    self.assertEqual(opt_in[0][1]["used"].shape, (60,))
    self.assertEqual(opt_in[0][1]["used"].host_transpose, (0, 3, 1, 2))

  def test_static_argnums_falls_back_gracefully(self):
    jitted_fn = jax.jit(lambda x, scale: x * scale, static_argnums=(1,))
    shlo_fn = main_lib.convert(
        jitted_fn,
        (jax.ShapeDtypeStruct((2, 4), jnp.float32), 2.0),
        {},
        enable_input_r1_optimization=True,
    )
    self.assertIsNone(shlo_fn.optimized_input_signature)

  def test_mixed_symbolic_and_static_inputs_skip_export_optimization(self):
    sym_b, = jax_export.symbolic_shape("b")
    params_spec = {"w": jax.ShapeDtypeStruct((2,), jnp.float32)}
    inputs_spec = {
        "dyn": jax.ShapeDtypeStruct((sym_b, 2), jnp.float32),
        "static_img": jax.ShapeDtypeStruct((2, 3, 5, 2), jnp.float32),
    }

    shlo_fn = main_lib.convert(
        lambda p, inp: {
            "a": inp["dyn"] * p["w"],
            "b": inp["static_img"] * p["w"],
        },
        (params_spec, inputs_spec),
        {},
        model_param_names=["w"],
        enable_input_r1_optimization=True,
        enable_output_r1_optimization=True,
        max_transpose_channels=3,
    )

    # Unconcretized shape polymorphism defers optimization to serving load time.
    self.assertIsNone(shlo_fn.optimized_input_signature)
    self.assertIsNone(shlo_fn.optimized_output_signature)

  def test_kwargs_spec_convert_optimizes_keyword_inputs(self):
    shlo_with_kwargs = main_lib.convert(
        lambda params, *, x: x * params["w"],
        ({"w": jax.ShapeDtypeStruct((2,), jnp.float32)},),
        {"x": jax.ShapeDtypeStruct((2, 3, 5, 2), jnp.float32)},
        model_param_names=["w"],
        enable_input_r1_optimization=True,
        max_transpose_channels=3,
    )
    opt_in_kw = shlo_with_kwargs.optimized_input_signature
    self.assertIsNotNone(opt_in_kw)
    self.assertEqual(opt_in_kw[0][0]["w"].shape, (2,))
    self.assertEqual(opt_in_kw[1]["x"].shape, (60,))
    self.assertEqual(opt_in_kw[1]["x"].host_transpose, (0, 3, 1, 2))

  def test_weightless_convert_optimizes_single_positional_input(self):
    shlo_weightless = main_lib.convert(
        lambda x: x * 2.0,
        (jax.ShapeDtypeStruct((2, 3, 5, 2), jnp.float32),),
        {},
        enable_input_r1_optimization=True,
        enable_output_r1_optimization=True,
        max_transpose_channels=3,
    )
    opt_in_wl = shlo_weightless.optimized_input_signature
    opt_out_wl = shlo_weightless.optimized_output_signature
    self.assertIsNotNone(opt_in_wl)
    self.assertIsNotNone(opt_out_wl)
    self.assertEqual(opt_in_wl[0][0].shape, (60,))
    self.assertEqual(opt_out_wl.shape, (60,))

  def test_input_aliasing_works_with_input_r1_and_nchw(self):
    bundle = typing.ApplyFnBundle(
        apply_fn=lambda p, x: x * p["w"],
        use_input_aliasing=True,
        enable_input_r1_optimization=True,
        enable_output_r1_optimization=False,
        max_transpose_channels=3,
    )
    shlo_fns = main_lib.convert_apply_fn_bundle_to_shlo(
        bundle,
        params={"w": jnp.ones((2,), dtype=jnp.float32)},
        input_signature=jax.ShapeDtypeStruct((2, 3, 5, 2), jnp.float32),
    )
    self.assertLen(shlo_fns, 1)
    shlo_fn = shlo_fns[0]
    self.assertEqual(shlo_fn.num_input_aliased_output, 1)
    opt_in = shlo_fn.optimized_input_signature
    self.assertIsNotNone(opt_in)
    # Arg 0: weight (2,); Arg 1: user input flattened to (60,) with NCHW;
    # Arg 2: donated output buffer kept at logical output shape (2, 3, 5, 2).
    self.assertEqual(opt_in[0][0]["w"].shape, (2,))
    self.assertEqual(opt_in[0][1].shape, (60,))
    self.assertEqual(opt_in[0][1].host_transpose, (0, 3, 1, 2))
    self.assertEqual(opt_in[0][2][0].shape, (2, 3, 5, 2))
    self.assertIsNone(opt_in[0][2][0].host_transpose)

  def test_pinned_host_memory_kind_preserves_host_placement(self):
    devices = mesh_utils.create_device_mesh((2,), devices=jax.devices()[:2])
    mesh = jax_sharding.Mesh(devices, ("b",))
    host_sharding = jax_sharding.NamedSharding(
        mesh, jax_sharding.PartitionSpec("b", None), memory_kind="pinned_host"
    )
    jitted_fn = jax.jit(
        lambda p, x: x + p["w"],
        in_shardings=(
            {
                "w": jax_sharding.NamedSharding(
                    mesh, jax_sharding.PartitionSpec()
                )
            },
            host_sharding,
        ),
    )
    shlo_fn = main_lib.convert(
        jitted_fn,
        (
            {"w": jax.ShapeDtypeStruct((4,), jnp.float32)},
            jax.ShapeDtypeStruct((2, 4), jnp.float32),
        ),
        {},
        model_param_names=["w"],
        enable_input_r1_optimization=True,
    )
    opt_in_sig = shlo_fn.optimized_input_signature
    self.assertIsNotNone(opt_in_sig)
    self.assertEqual(opt_in_sig[0][1].shape, (8,))
    self.assertEqual(opt_in_sig[0][1].placement, type_pb2.HOST)

  @parameterized.named_parameters(
      ("c_eq_max_and_lt_w_eligible", (1, 3, 5, 3), 3, True),
      ("c_gt_max_ineligible", (1, 3, 5, 4), 3, False),
      ("c_eq_w_ineligible", (1, 3, 3, 3), 3, False),
      ("c_gt_w_ineligible", (1, 5, 2, 3), 3, False),
      ("zero_channel_ineligible", (1, 3, 5, 0), 3, False),
  )
  def test_nchw_channel_and_width_boundaries(
      self, shape: tuple[int, ...], max_channels: int, expect_transpose: bool
  ):
    shlo_fn = main_lib.convert(
        lambda x: x + 1.0,
        (jax.ShapeDtypeStruct(shape, jnp.float32),),
        {},
        max_transpose_channels=max_channels,
    )
    opt_in_sig = shlo_fn.optimized_input_signature
    if expect_transpose:
      self.assertIsNotNone(opt_in_sig)
      self.assertEqual(opt_in_sig[0][0].host_transpose, (0, 3, 1, 2))
    else:
      self.assertIsNone(opt_in_sig)

  @parameterized.named_parameters(
      ("bfloat16_eligible", jnp.bfloat16, True),
      ("float8_e4m3fn_eligible", jnp.float8_e4m3fn, True),
      ("int8_eligible", jnp.int8, True),
      ("int4_ineligible", jnp.int4, False),
      ("uint4_ineligible", jnp.uint4, False),
      ("bool_ineligible", jnp.bool_, False),
      ("complex64_ineligible", jnp.complex64, False),
      ("float0_ineligible", jax.dtypes.float0, False),
  )
  def test_nchw_dtype_eligibility_matches_cpp_bitwidths(
      self, dtype, expect_eligible: bool
  ):
    shlo_fn = main_lib.convert(
        jax.jit(lambda x: x),
        (jax.ShapeDtypeStruct((1, 3, 5, 2), dtype),),
        {},
        max_transpose_channels=3,
    )
    opt_in_sig = shlo_fn.optimized_input_signature
    if expect_eligible:
      self.assertIsNotNone(opt_in_sig)
      self.assertEqual(opt_in_sig[0][0].host_transpose, (0, 3, 1, 2))
    else:
      self.assertIsNone(opt_in_sig)

  def test_explicit_gspmd_sharding_skips_optimization_without_rank_mismatch(
      self,
  ):
    devices = jax.devices()[:2]
    mesh = jax_sharding.Mesh(np.array(devices), ("b",))
    named_4d = jax_sharding.NamedSharding(
        mesh, jax_sharding.PartitionSpec("b", None, None, None)
    )
    gspmd_4d = jex.sharding.GSPMDSharding(
        devices, named_4d._to_xla_hlo_sharding(4)
    )
    jitted_fn = jax.jit(
        lambda p, inputs: inputs["gspmd"] * p["w"] + inputs["plain"],
        in_shardings=(
            {
                "w": jax_sharding.NamedSharding(
                    mesh, jax_sharding.PartitionSpec()
                )
            },
            {"gspmd": gspmd_4d, "plain": named_4d},
        ),
        out_shardings=gspmd_4d,
    )
    params_spec = {"w": jax.ShapeDtypeStruct((1, 1, 1, 2), jnp.float32)}
    inputs_spec = {
        "gspmd": jax.ShapeDtypeStruct((2, 3, 5, 2), jnp.float32),
        "plain": jax.ShapeDtypeStruct((2, 3, 5, 2), jnp.float32),
    }

    shlo_fn = main_lib.convert(
        jitted_fn,
        (params_spec, inputs_spec),
        {},
        model_param_names=["w"],
        enable_input_r1_optimization=True,
        enable_output_r1_optimization=True,
        max_transpose_channels=3,
    )

    opt_in_sig = shlo_fn.optimized_input_signature
    self.assertIsNotNone(opt_in_sig)
    # Explicit `GSPMDSharding` leaf is skipped cleanly without rank mismatch;
    # `NamedSharding` batch-sharded leaf is still NCHW + R1 optimized.
    self.assertEqual(opt_in_sig[0][1]["gspmd"].shape, (2, 3, 5, 2))
    self.assertIsNone(opt_in_sig[0][1]["gspmd"].host_transpose)
    self.assertEqual(opt_in_sig[0][1]["plain"].shape, (60,))
    self.assertEqual(opt_in_sig[0][1]["plain"].host_transpose, (0, 3, 1, 2))
    self.assertIsNone(shlo_fn.optimized_output_signature)

  def test_convert_apply_fn_bundle_rejects_output_r1_with_input_aliasing(self):
    bundle = typing.ApplyFnBundle(
        apply_fn=lambda p, x: x,
        use_input_aliasing=True,
        enable_output_r1_optimization=True,
    )
    with self.assertRaisesRegex(
        ValueError, "Output R1 optimization is incompatible"
    ):
      main_lib.convert_apply_fn_bundle_to_shlo(
          bundle,
          params={},
          input_signature=shlo_type.ShloTensorSpec(
              shape=(2, 4), dtype=shlo_type.ShloDType.f32
          ),
      )


if __name__ == "__main__":
  absltest.main()
