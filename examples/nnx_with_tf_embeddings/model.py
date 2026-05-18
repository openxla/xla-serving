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

"""Model for NNX with TF TPUEmbeddingForServing layer for embeddings."""

from flax import nnx
import tensorflow as tf

from examples.nnx import model as nnx_model

EmbeddingSpec = nnx_model.EmbeddingSpec
Tower = nnx_model.Tower
Model = nnx_model.Model


def create_embedding_table_configs(
    table_count: int,
) -> dict[str, tf.tpu.experimental.embedding.TableConfig]:
  """Creates a dictionary of TableConfig for TPU embeddings.

  Args:
    table_count: The number of embedding tables to create.

  Returns:
    A dictionary mapping table names to
    `tf.tpu.experimental.embedding.TableConfig`
    instances.
  """
  vocab_sizes = [2, 25475, 48994, 92232, 100000, 150000]
  embedding_dims = [16, 32, 64, 256]
  table_configs = {}
  for i in range(table_count):
    vocab_size = vocab_sizes[i % len(vocab_sizes)]
    embedding_dim = embedding_dims[i % len(embedding_dims)]
    table_name = f"table_{i+1}"
    table_config = tf.tpu.experimental.embedding.TableConfig(
        vocabulary_size=vocab_size,
        dim=embedding_dim,
        name=table_name,
        initializer=tf.random_uniform_initializer(minval=-0.05, maxval=0.05),
    )
    table_configs[table_name] = table_config
  return table_configs


def create_embedding_feature_configs(
    feature_names: list[str],
    table_configs: dict[str, tf.tpu.experimental.embedding.TableConfig],
    feature_to_table_names: dict[str, str],
) -> dict[str, tf.tpu.experimental.embedding.FeatureConfig]:
  """Creates a preprocessor for embedding lookups."""
  feature_configs = {}
  for feature_name in feature_names:
    table_name = feature_to_table_names[feature_name]
    table_config = table_configs[table_name]
    feature_config = tf.tpu.experimental.embedding.FeatureConfig(
        table=table_config,
        name=feature_name,
    )
    feature_configs[feature_name] = feature_config
  return feature_configs


def create_request_embedding_feature_configs(
    num_request_features: int,
    table_configs: dict[str, tf.tpu.experimental.embedding.TableConfig],
) -> dict[str, tf.tpu.experimental.embedding.FeatureConfig]:
  """Creates a preprocessor for embedding lookups."""
  if num_request_features > len(table_configs):
    raise ValueError(
        f"num_request_features ({num_request_features}) must be less than or"
        f" equal to the number of table configs ({len(table_configs)})."
    )
  feature_names = [
      f"request_feature_{i+1}" for i in range(num_request_features)
  ]
  feature_to_table_names = {}
  for i, feature_name in enumerate(feature_names):
    table_name = f"table_{i+1}"
    feature_to_table_names[feature_name] = table_name
  return create_embedding_feature_configs(
      feature_names=feature_names,
      table_configs=table_configs,
      feature_to_table_names=feature_to_table_names,
  )


def create_candidate_embedding_feature_configs(
    num_candidate_features: int,
    num_request_features: int,
    table_configs: dict[str, tf.tpu.experimental.embedding.TableConfig],
) -> dict[str, tf.tpu.experimental.embedding.FeatureConfig]:
  """Creates a preprocessor for embedding lookups."""
  if num_candidate_features + num_request_features > len(table_configs):
    raise ValueError(
        "num_candidate_features + num_request_features"
        f" ({num_candidate_features} + {num_request_features}) must be less"
        f" than or equal to the number of table configs ({len(table_configs)})."
    )
  feature_names = [
      f"candidate_feature_{i+1}" for i in range(num_candidate_features)
  ]
  feature_to_table_names = {}
  for i, feature_name in enumerate(feature_names):
    table_name = f"table_{i+ num_request_features + 1}"
    feature_to_table_names[feature_name] = table_name
  return create_embedding_feature_configs(
      feature_names=feature_names,
      table_configs=table_configs,
      feature_to_table_names=feature_to_table_names,
  )


def to_embedding_specs(
    feature_configs: dict[str, tf.tpu.experimental.embedding.FeatureConfig],
) -> list[EmbeddingSpec]:
  """Converts feature configs to embedding specs."""
  embedding_specs = []
  for _, feature_config in feature_configs.items():
    embedding_specs.append(
        EmbeddingSpec(
            name=feature_config.name,
            vocab_size=feature_config.table.vocabulary_size,
            embedding_dim=feature_config.table.dim,
        )
    )
  return embedding_specs


NUM_PREDICTION_TYPES = 1


def build_model(
    rngs: nnx.Rngs,
    num_request_features: int,
    num_candidate_features: int,
) -> tuple[tf.tpu.experimental.embedding.TPUEmbeddingForServing, Model]:
  """Build the recommender model with two towers and a prediction layer."""
  table_count = num_request_features + num_candidate_features
  table_configs = create_embedding_table_configs(table_count)

  request_feature_configs = create_request_embedding_feature_configs(
      num_request_features, table_configs
  )
  candidate_feature_configs = create_candidate_embedding_feature_configs(
      num_candidate_features, num_request_features, table_configs
  )
  request_embedding_specs = to_embedding_specs(request_feature_configs)
  candidate_embedding_specs = to_embedding_specs(candidate_feature_configs)
  request_input_dim = sum(fs.embedding_dim for fs in request_embedding_specs)
  candidate_input_dim = sum(
      fs.embedding_dim for fs in candidate_embedding_specs
  )

  request_tower = Tower(
      embeddings=request_embedding_specs,
      layer=nnx.Sequential(
          nnx.Linear(request_input_dim, 2048, rngs=rngs),
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
      embeddings=candidate_embedding_specs,
      layer=nnx.Sequential(
          nnx.Linear(candidate_input_dim, 2048, rngs=rngs),
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

  feature_configs = request_feature_configs | candidate_feature_configs

  embedding_layer = tf.tpu.experimental.embedding.TPUEmbeddingForServing(
      feature_config=feature_configs,
      optimizer=None,
  )
  embedding_layer.build()

  model = Model(
      request_tower=request_tower,
      candidate_tower=candidate_tower,
      prediction_layer=prediction_layer,
  )

  return embedding_layer, model
