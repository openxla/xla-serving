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

"""A recommender model with two towers and a prediction layer."""

from flax import nnx
import tensorflow as tf

from .third_party.neptune.examples.nnx import model as nnx_model

EmbeddingSpec = nnx_model.EmbeddingSpec
Tower = nnx_model.Tower
Model = nnx_model.Model


REQUEST_EMBEDDINGS = [
    EmbeddingSpec(
        name="feature_1",
        vocab_size=2,
        embedding_dim=16,
    ),
    EmbeddingSpec(
        name="feature_2",
        vocab_size=92232,
        embedding_dim=32,
    ),
]

CANDIDATE_EMBEDDINGS = [
    EmbeddingSpec(
        name="feature_3",
        vocab_size=25475,
        embedding_dim=128,
    ),
    EmbeddingSpec(
        name="feature_4",
        vocab_size=48994,
        embedding_dim=64,
    ),
]


ALL_FEATURES = REQUEST_EMBEDDINGS + CANDIDATE_EMBEDDINGS
REQUEST_INPUT_DIM = sum(fs.embedding_dim for fs in REQUEST_EMBEDDINGS)
CANDIDATE_INPUT_DIM = sum(fs.embedding_dim for fs in CANDIDATE_EMBEDDINGS)
NUM_PREDICTION_TYPES = 1


class DenseEmbedding(tf.keras.layers.Layer):
  """Dense embedding tables using standard tf.keras.layers.Embedding modules."""

  def __init__(
      self,
      embedding_specs: list[EmbeddingSpec],
      **kwargs,
  ):
    super().__init__(**kwargs)
    self._embeddings_by_name = {}

    for spec in embedding_specs:
      self._embeddings_by_name[spec.name] = tf.keras.layers.Embedding(
          input_dim=spec.vocab_size,
          output_dim=spec.embedding_dim,
          embeddings_initializer=tf.keras.initializers.RandomUniform(),
          name=f"{spec.name}_embedding",
      )

  def call(self, inputs: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    outputs = {}
    for embedding_name, ids in inputs.items():
      if embedding_name not in self._embeddings_by_name:
        raise ValueError(
            f"Feature '{embedding_name}' not found in embedding layer"
        )
      embedding_table = self._embeddings_by_name[embedding_name]
      outputs[embedding_name] = embedding_table(ids)

    return outputs


class EmbeddingLayer(tf.keras.layers.Layer):
  """The embedding layer implementation using tf.keras.layers.Layer."""

  def __init__(
      self,
      embedding_specs: list[EmbeddingSpec],
      **kwargs,
  ):
    super().__init__(**kwargs)
    self._embedding_names = [emb.name for emb in embedding_specs]
    self.embedding = DenseEmbedding(
        embedding_specs=embedding_specs,
    )

  def call(self, inputs: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    embedding_inputs = {name: inputs[name] for name in self._embedding_names}
    return self.embedding(embedding_inputs)


def build_model(
    rngs: nnx.Rngs,
) -> tuple[EmbeddingLayer, Model]:
  """Build the recommender model with two towers and a prediction layer."""
  request_tower = Tower(
      embeddings=REQUEST_EMBEDDINGS,
      layer=nnx.Sequential(
          nnx.Linear(REQUEST_INPUT_DIM, 2048, rngs=rngs),
          nnx.LayerNorm(2048, rngs=rngs),
          nnx.silu,
          nnx.Linear(2048, 2048, rngs=rngs),
          nnx.LayerNorm(2048, rngs=rngs),
          nnx.silu,
          nnx.Linear(2048, 1024, rngs=rngs),
          nnx.LayerNorm(1024, rngs=rngs),
          nnx.silu,
          nnx.Linear(1024, 384, rngs=rngs),
      ),
  )

  candidate_tower = Tower(
      embeddings=CANDIDATE_EMBEDDINGS,
      layer=nnx.Sequential(
          nnx.Linear(CANDIDATE_INPUT_DIM, 2048, rngs=rngs),
          nnx.LayerNorm(2048, rngs=rngs),
          nnx.silu,
          nnx.Linear(2048, 2048, rngs=rngs),
          nnx.LayerNorm(2048, rngs=rngs),
          nnx.silu,
          nnx.Linear(2048, 1024, rngs=rngs),
          nnx.LayerNorm(1024, rngs=rngs),
          nnx.silu,
          nnx.Linear(1024, 384, rngs=rngs),
      ),
  )

  prediction_layer = nnx.Linear(384, NUM_PREDICTION_TYPES, rngs=rngs)

  embedding_layer = EmbeddingLayer(
      embedding_specs=ALL_FEATURES,
  )

  model = Model(
      request_tower=request_tower,
      candidate_tower=candidate_tower,
      prediction_layer=prediction_layer,
  )

  return embedding_layer, model
