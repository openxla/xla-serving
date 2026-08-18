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

import dataclasses
from flax import nnx
import jax


@dataclasses.dataclass
class EmbeddingSpec:
  """Specification for an embedding table."""

  name: str
  vocab_size: int
  embedding_dim: int


class DenseEmbedding(nnx.Module):
  """Dense embedding tables using standard nnx.Embed modules."""

  def __init__(
      self,
      embedding_specs: list[EmbeddingSpec],
      rngs: nnx.Rngs,
  ):
    self._embeddings_by_name: nnx.Dict[str, nnx.Embed] = nnx.Dict()  # pyrefly: ignore[bad-specialization]

    for spec in embedding_specs:
      self._embeddings_by_name[spec.name] = nnx.Embed(
          num_embeddings=spec.vocab_size,
          features=spec.embedding_dim,
          rngs=rngs,
          embedding_init=jax.nn.initializers.uniform(),
          embedding_metadata={
              "is_embedding": True,
          },
      )

  def __call__(self, inputs: dict[str, jax.Array]) -> dict[str, jax.Array]:
    outputs = {}
    for embedding_name, ids in inputs.items():
      if embedding_name not in self._embeddings_by_name:
        raise ValueError(
            f"Feature '{embedding_name}' not found in embedding layer"
        )
      embedding_table = self._embeddings_by_name[embedding_name]
      outputs[embedding_name] = embedding_table(ids)  # pyrefly: ignore[not-callable]

    return outputs


class EmbeddingLayer(nnx.Module):
  """The embedding layer implementation."""

  def __init__(
      self,
      embedding_specs: list[EmbeddingSpec],
      rngs: nnx.Rngs,
  ):
    self._embedding_names = [emb.name for emb in embedding_specs]
    self.embedding = DenseEmbedding(
        embedding_specs=embedding_specs,
        rngs=rngs,
    )

  def __call__(self, inputs: dict[str, jax.Array]) -> dict[str, jax.Array]:
    embedding_inputs = {name: inputs[name] for name in self._embedding_names}
    return self.embedding(embedding_inputs)


class Tower(nnx.Module):
  """A tower implementation with embedding lookups."""

  def __init__(
      self,
      embeddings: list[EmbeddingSpec],
      layer: nnx.Module,
      name: str | None = None,
  ):
    self.name = name if name is not None else "tower"
    self.embeddings = sorted(embeddings, key=lambda fs: fs.name)
    self.layer = layer

  def __call__(self, embeddings: dict[str, jax.Array]) -> jax.Array:
    embedding_inputs = [embeddings[fs.name] for fs in self.embeddings]
    concatenated_inputs = jax.numpy.concatenate(embedding_inputs, axis=-1)
    return self.layer(concatenated_inputs)


class Model(nnx.Module):
  """The model implementation."""

  def __init__(
      self,
      request_tower: nnx.Module,
      candidate_tower: nnx.Module,
      prediction_layer: nnx.Module,
      embedding_layer: EmbeddingLayer | None = None,
  ):
    self.request_tower = request_tower
    self.candidate_tower = candidate_tower
    self.prediction_layer = prediction_layer
    self.embedding_layer = embedding_layer

  def _logits(self, inputs: nnx.Dict[str, jax.Array]) -> jax.Array:  # pyrefly: ignore[bad-specialization]
    embeddings = (
        self.embedding_layer(inputs)  # pyrefly: ignore[bad-argument-type]
        if self.embedding_layer is not None
        else inputs
    )
    request_output = self.request_tower(embeddings)
    candidate_output = self.candidate_tower(embeddings)
    merged_output = request_output * candidate_output
    return self.prediction_layer(merged_output)

  def __call__(self, inputs: nnx.Dict[str, jax.Array]) -> dict[str, jax.Array]:  # pyrefly: ignore[bad-specialization]
    return {"pred": jax.nn.sigmoid(self._logits(inputs))}


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


def build_model(
    rngs: nnx.Rngs,
    *,
    separate_embedding_layer: bool = False,
) -> tuple[EmbeddingLayer, Model]:
  """Build the recommender model with two towers and a prediction layer."""
  embedding_layer = EmbeddingLayer(
      embedding_specs=ALL_FEATURES,
      rngs=rngs,
  )

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

  if separate_embedding_layer:
    model_embedding_layer = None
  else:
    model_embedding_layer = embedding_layer

  model = Model(
      embedding_layer=model_embedding_layer,
      request_tower=request_tower,
      candidate_tower=candidate_tower,
      prediction_layer=prediction_layer,
  )

  return embedding_layer, model
