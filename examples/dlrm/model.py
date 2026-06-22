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


"""DLRM DCN v2 model."""

from typing import List, Mapping

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

# Taken as is from model training SC configuration, reduced some of the vocab
# sizes for testing.

VOCAB_SIZES = [
    400000,
    39060,
    17295,
    7424,
    20265,
    3,
    7122,
    1543,
    63,
    400000,
    3067956,
    405282,
    10,
    2209,
    11938,
    155,
    4,
    976,
    14,
    400000,
    400000,
    400000,
    590152,
    12973,
    108,
    36,
]
VALENCY = [
    3,
    2,
    1,
    2,
    6,
    1,
    1,
    1,
    1,
    7,
    3,
    8,
    1,
    6,
    9,
    5,
    1,
    1,
    1,
    12,
    100,
    27,
    10,
    3,
    1,
    1,
]

_EMBEDDING_SIZE = 128
_BATCH_SIZE = 64


def uniform_init(bound: float):
  def init(key, shape, dtype=jnp.float_):
    return jax.random.uniform(
        key, shape=shape, dtype=dtype, minval=-bound, maxval=bound
    )

  return init


class DLRMDCNV2(nn.Module):
  """DLRM DCN v2 model."""
  global_batch_size: int
  embedding_size: int
  bottom_mlp_dims: List[int]
  top_mlp_dims = [1024, 1024, 512, 256, 1]
  dcn_layers: int = 3
  projection_dim: int = 512

  def bottom_mlp(self, x):
    for dim in self.bottom_mlp_dims:
      previous_dim = x.shape[-1]
      bound = np.sqrt(1.0 / previous_dim)
      x = nn.Dense(
          dim,
          kernel_init=uniform_init(bound),
          bias_init=uniform_init(bound),
      )(x)
      x = nn.relu(x)
    return x

  def top_mlp(self, x):
    previous_dim = x.shape[-1]
    for dim in self.top_mlp_dims[:-1]:
      bound = np.sqrt(1.0 / previous_dim)
      x = nn.Dense(
          dim,
          kernel_init=uniform_init(bound),
          bias_init=uniform_init(bound),
      )(x)
      x = nn.relu(x)
      previous_dim = dim

    bound = np.sqrt(1.0 / previous_dim)
    x = nn.Dense(
        self.top_mlp_dims[-1],
        kernel_init=uniform_init(bound),
        bias_init=uniform_init(bound),
    )(x)
    x = nn.sigmoid(x)
    return x

  def dcn_layer(self, x0):
    xl = x0
    input_dim = x0.shape[-1]

    for i in range(self.dcn_layers):
      u_kernel = self.param(
          f"u_kernel_{i}",
          nn.initializers.xavier_normal(),
          (input_dim, self.projection_dim),
      )
      v_kernel = self.param(
          f"v_kernel_{i}",
          nn.initializers.xavier_normal(),
          (self.projection_dim, input_dim),
      )
      bias = self.param(f"bias_{i}", nn.initializers.zeros, (input_dim,))

      u_output = jnp.matmul(xl, u_kernel)
      v_output = jnp.matmul(u_output, v_kernel)
      v_output += bias

      xl = x0 * v_output + xl

    return xl

  @nn.compact
  def __call__(self, dense_features, sparse_embeddings):
    dense_outputs = self.bottom_mlp(dense_features)

    sparse_embeddings = jax.tree.flatten(sparse_embeddings)
    concatenated_embeddings = jnp.concatenate(sparse_embeddings[0], axis=1)
    batch_size = dense_outputs.shape[0]
    # Concatenate dense features and embeddings. We're using global batch size
    # here because we're doing global view of the data in the training loop.
    interaction_args = jax.lax.concatenate(
        [
            dense_outputs.reshape((batch_size, 1, self.embedding_size)),
            concatenated_embeddings.reshape(
                (batch_size, 26, self.embedding_size)
            ),
        ],
        dimension=1,
    )
    interaction_args = interaction_args.reshape((batch_size, -1))
    interaction_outputs = self.dcn_layer(interaction_args)
    predictions = self.top_mlp(interaction_outputs)
    predictions = jnp.reshape(predictions, (-1,))

    return predictions


def create_feature_specs(
    vocab_sizes: List[int],
) -> tuple[
    Mapping[str, tf.tpu.experimental.embedding.TableConfig],
    Mapping[str, tf.tpu.experimental.embedding.FeatureConfig],
]:
  """Creates the table and feature configs for the DLRM model."""
  table_configs = {}
  feature_configs = {}
  for i, vocab_size in enumerate(vocab_sizes):
    table_name = f"{i}"
    feature_name = f"{i}"
    table_config = tf.tpu.experimental.embedding.TableConfig(
        vocabulary_size=vocab_size,
        dim=_EMBEDDING_SIZE,
        name=table_name,
        combiner="sum",
        optimizer=None,
    )

    feature_config = tf.tpu.experimental.embedding.FeatureConfig(
        name=feature_name, table=table_config
    )
    feature_configs[feature_name] = feature_config
    table_configs[table_name] = table_config
  return table_configs, feature_configs


def build_model() -> (
    tuple[tf.tpu.experimental.embedding.TPUEmbeddingForServing, nn.Module]
):
  """Build the recommender model with two towers and a prediction layer."""
  _, feature_configs = create_feature_specs(VOCAB_SIZES)
  model = DLRMDCNV2(
      global_batch_size=_BATCH_SIZE,
      embedding_size=_EMBEDDING_SIZE,
      bottom_mlp_dims=[512, 256, 128],
      dcn_layers=3,
      projection_dim=512,
  )

  embedding_layer = tf.tpu.experimental.embedding.TPUEmbeddingForServing(
      feature_config=feature_configs,
      optimizer=None,
  )
  embedding_layer.build()

  return embedding_layer, model
