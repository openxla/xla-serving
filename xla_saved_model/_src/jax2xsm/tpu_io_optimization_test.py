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

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import sharding as jax_sharding
from jax.experimental import mesh_utils
import jax.numpy as jnp

from xla_saved_model._src.core import shlo_type
from xla_saved_model._src.jax2xsm import main_lib
from xla_saved_model._src.jax2xsm import tpu_io_optimization
from xla_saved_model._src.jax2xsm import typing

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"


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

    # Weight 'w' remains (4, 8); user input 'x' is flattened to (8,); 1-D input
    # 'bias_1d' remains (8,).
    self.assertEqual(shlo_fn.input_signature[0][0]["w"].shape, (4, 8))
    self.assertEqual(shlo_fn.input_signature[0][1]["x"].shape, (8,))
    self.assertEqual(shlo_fn.input_signature[0][1]["bias_1d"].shape, (8,))
    self.assertIsNotNone(shlo_fn.original_input_signature)
    self.assertEqual(shlo_fn.original_input_signature["x"].shape, (2, 4))
    self.assertEqual(shlo_fn.original_input_signature["bias_1d"].shape, (8,))
    self.assertIsNone(shlo_fn.original_output_signature)
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

    self.assertEqual(shlo_fn.output_signature["out_2d"].shape, (12,))
    self.assertEqual(shlo_fn.output_signature["out_1d"].shape, (2,))
    self.assertIsNone(shlo_fn.original_input_signature)
    self.assertIsNotNone(shlo_fn.original_output_signature)
    self.assertEqual(shlo_fn.original_output_signature["out_2d"].shape, (2, 6))
    self.assertEqual(shlo_fn.original_output_signature["out_1d"].shape, (2,))

  def test_nchw_transpose_and_input_r1_composition(self):
    params_spec = {"scale": jax.ShapeDtypeStruct((1, 4, 4, 2), jnp.float32)}
    inputs_spec = {
        # Eligible: 4-D, C=2 <= max_transpose_channels=3 and C=2 < W=4, f32.
        "img": jax.ShapeDtypeStruct((1, 4, 4, 2), jnp.float32),
        # Ineligible for NCHW (C=4 >= W=4), but eligible for R1.
        "wide_c": jax.ShapeDtypeStruct((1, 2, 4, 4), jnp.float32),
        # Ineligible for NCHW (bool dtype), but eligible for R1.
        "mask": jax.ShapeDtypeStruct((1, 4, 4, 2), jnp.bool_),
    }

    def apply_fn_mixed(params, inputs):
      return (
          inputs["img"] * params["scale"]
          + jnp.reshape(inputs["wide_c"], (1, 4, 4, 2))
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
        shlo_fn_nchw_only.input_signature[0][1]["img"].shape, (1, 2, 4, 4)
    )
    self.assertEqual(
        shlo_fn_nchw_only.input_signature[0][1]["img"].host_transpose,
        tpu_io_optimization.NHWC_TO_NCHW_PERM,
    )
    self.assertEqual(
        shlo_fn_nchw_only.original_input_signature["img"].shape, (1, 4, 4, 2)
    )
    self.assertEqual(
        shlo_fn_nchw_only.original_input_signature["img"].host_transpose,
        tpu_io_optimization.NHWC_TO_NCHW_PERM,
    )
    self.assertEqual(
        shlo_fn_nchw_only.input_signature[0][1]["wide_c"].shape, (1, 2, 4, 4)
    )
    self.assertIsNone(
        shlo_fn_nchw_only.input_signature[0][1]["wide_c"].host_transpose
    )
    self.assertIsNone(
        shlo_fn_nchw_only.input_signature[0][1]["mask"].host_transpose
    )

    # Now enable both NCHW transpose and Input R1 flattening.
    shlo_fn_both = main_lib.convert(
        apply_fn_mixed,
        (params_spec, inputs_spec),
        {},
        enable_input_r1_optimization=True,
        max_transpose_channels=3,
    )
    self.assertEqual(shlo_fn_both.input_signature[0][1]["img"].shape, (32,))
    self.assertEqual(
        shlo_fn_both.input_signature[0][1]["img"].host_transpose,
        tpu_io_optimization.NHWC_TO_NCHW_PERM,
    )
    self.assertEqual(
        shlo_fn_both.original_input_signature["img"].shape, (1, 4, 4, 2)
    )
    self.assertEqual(
        shlo_fn_both.original_input_signature["img"].host_transpose,
        tpu_io_optimization.NHWC_TO_NCHW_PERM,
    )
    self.assertEqual(shlo_fn_both.input_signature[0][1]["wide_c"].shape, (32,))
    self.assertIsNone(
        shlo_fn_both.input_signature[0][1]["wide_c"].host_transpose
    )

  def test_sharding_collapses_batch_and_skips_non_batch(self):
    devices = mesh_utils.create_device_mesh((4, 2))
    mesh = jax_sharding.Mesh(devices, ("b", "m"))

    batch_sharded = jax_sharding.NamedSharding(
        mesh, jax_sharding.PartitionSpec("b", None)
    )
    non_batch_sharded = jax_sharding.NamedSharding(
        mesh, jax_sharding.PartitionSpec("b", "m")
    )

    def apply_fn(params, inputs):
      return inputs["batch_only"] + inputs["non_batch"] + params["w"]

    jitted_fn = jax.jit(
        apply_fn,
        in_shardings=(
            {"w": non_batch_sharded},
            {"batch_only": batch_sharded, "non_batch": non_batch_sharded},
        ),
        out_shardings=batch_sharded,
    )

    params_spec = {"w": jax.ShapeDtypeStruct((4, 2), jnp.float32)}
    inputs_spec = {
        "batch_only": jax.ShapeDtypeStruct((4, 2), jnp.float32),
        "non_batch": jax.ShapeDtypeStruct((4, 2), jnp.float32),
    }

    shlo_fn = main_lib.convert(
        jitted_fn,
        (params_spec, inputs_spec),
        {},
        enable_input_r1_optimization=True,
        enable_output_r1_optimization=True,
    )

    # 'batch_only' is flattened to (8,) with 1-D batch sharding; 'non_batch' is
    # skipped and remains (4, 2).
    self.assertEqual(shlo_fn.input_signature[0][1]["batch_only"].shape, (8,))
    self.assertEqual(
        list(
            shlo_fn.input_signature[0][1][
                "batch_only"
            ].sharding.tile_assignment_dimensions
        ),
        [4, 2],
    )
    self.assertEqual(shlo_fn.input_signature[0][1]["non_batch"].shape, (4, 2))
    self.assertEqual(shlo_fn.output_signature.shape, (8,))
    self.assertEqual(shlo_fn.original_output_signature.shape, (4, 2))

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
