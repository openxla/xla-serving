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

"""Unit tests for sampling utilities."""

from absl.testing import absltest
import jax
import jax.numpy as jnp

from neptune_model._src.llm import sampling


class SamplingTest(absltest.TestCase):

  def test_top_k_sample_direct(self):
    key = jax.random.PRNGKey(0)
    logits = jnp.array([1.0, 5.0, 100.0], dtype=jnp.float32)
    sampled = sampling.top_k_sample(key, logits, k=1, temperature=1.0)
    self.assertEqual(int(sampled), 2)

  def test_greedy_sample_direct(self):
    logits = jnp.array([10.0, 500.0, 50.0], dtype=jnp.float32)
    sampled = sampling.greedy_sample(logits)
    self.assertEqual(int(sampled), 1)


if __name__ == "__main__":
  absltest.main()
