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

"""Python utilities for block-wise integer quantization."""

import numpy as np


def block_quantize(
    weights: np.ndarray, block_size: int, symmetric: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Quantizes 2D weights array to 8-bit integer block-wise format.

  Args:
    weights: 2D float32 numpy array of shape [VocabSize, Dim].
    block_size: Size of each block along the embedding dimension.
    symmetric: If True, uses symmetric quantization (no offsets).

  Returns:
    quantized_data: int8 array of shape [VocabSize, Dim].
    scales: float32 array of shape [VocabSize, NumBlocks].
    offsets: float32 array of shape [VocabSize, NumBlocks] (empty if symmetric).
  """
  if weights.ndim != 2:
    raise ValueError(f"Weights must be a 2D array. Got shape: {weights.shape}")
  if block_size <= 0:
    raise ValueError("Block size must be positive.")

  vocab_size, dim = weights.shape
  num_blocks = (dim + block_size - 1) // block_size

  quantized_data = np.zeros_like(weights, dtype=np.int8)
  scales = np.zeros((vocab_size, num_blocks), dtype=np.float32)
  offsets = (
      np.zeros((0,), dtype=np.float32)
      if symmetric
      else np.zeros((vocab_size, num_blocks), dtype=np.float32)
  )

  for i in range(vocab_size):
    row = weights[i]
    for b_idx in range(num_blocks):
      start = b_idx * block_size
      end = min(start + block_size, dim)
      block = row[start:end]

      if symmetric:
        max_val = np.max(np.abs(block))
        scale = max(max_val / 127.0, 1e-8)
        q_block = np.round(block / scale)
        q_block = np.clip(q_block, -128, 127)
        quantized_data[i, start:end] = q_block.astype(np.int8)
        scales[i, b_idx] = scale
      else:
        min_val = np.min(block)
        max_val = np.max(block)
        scale = max((max_val - min_val) / 255.0, 1e-8)
        offset = min_val + 128.0 * scale
        q_block = np.round((block - offset) / scale)
        q_block = np.clip(q_block, -128, 127)
        quantized_data[i, start:end] = q_block.astype(np.int8)
        scales[i, b_idx] = scale
        offsets[i, b_idx] = offset

  return quantized_data, scales, offsets
