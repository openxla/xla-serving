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

"""Exports the DLRM-DCNv2 benchmark model with a *TensorFlow* preprocessor.

Counterpart of `export_benchmark_model.py` for the "CompositeProcessor vs. TF"
comparison. It loads the weights that binary wrote to `--baseline_weights_dir`
(`tables.npz`, `params.npz`, `config.json`) -- so every arm of the benchmark
serves bit-identical weights and `parity_check.py` keeps working unchanged --
and writes XSE exports whose CPU preprocessor is a TF SavedModel function
instead of the JCE `CompositeProcessor`:

* `--batched_export_path`:   server-side `BatchOptions` (same configuration as
                             the JCE export: `PRE_PROCESSOR_AND_MODEL_FUNCTION`,
                             PAD_UP to the same buckets).
* `--unbatched_export_path`: `NO_BATCHING`.

The accelerator part (DCNv2 as a `JaxModel`) is exported exactly as in the JCE
arm; the TF preprocessor hands it the same 27 tensors (26 pooled `[B, D]`
embeddings + the dense passthrough), so the only difference between the arms
is the embedding lookup implementation.

Two TF lookup implementations are available (`--tf_lookup`):

* `fused` (default): the production-grade TF path. For a multi-valent feature
  `tf.sparse.segment_sum(table, ids, segment_ids)` -- a single fused
  `SparseSegmentSum[WithNumSegments]` kernel that reads the rows and sums them
  into the output without materialising a `[B, V, D]` intermediate. This is
  the op graph `tf.nn.embedding_lookup_sparse(..., combiner='sum',
  allow_fast_lookup=True)` / `tf.nn.safe_embedding_lookup_sparse` (and hence
  `tf.feature_column.embedding_column` and Keras' sparse-input `Embedding`)
  emit, minus the SparseTensor plumbing that dense fixed-valency ids do not
  need. Single-valent features are a plain `tf.gather`.
* `gather`: `tf.reduce_sum(tf.gather(table, ids), axis=1)` -- the
  `ResourceGather` + `Sum` pair that `tf.nn.embedding_lookup` on a dense id
  tensor, `tf.tpu.experimental.embedding.TPUEmbeddingForServing` (what
  `examples/dlrm/export.py` uses) and Keras' dense-input `Embedding` layer
  produce. Materialises `[B, V, D]` per feature (up to ~210 MB for the
  valency-100 feature at B=4096 and D=128) before reducing it.

Both are exported so that the TF side cannot be called a straw man: `fused`
minimises memory traffic, `gather` is what most existing TF recommender
exports actually contain. The server runs the TF function through TFRT
(`--define=ENABLE_TF=1` builds / `*-with-tf` images) with the production
defaults (MLRT, invariant `ReadVariableOp`s hoisted out of the per-request
graph).
"""

from __future__ import annotations

import dataclasses
import functools
import os
import time
from typing import Any, Mapping, Sequence

from absl import app
from absl import flags
from absl import logging
import jax
import numpy as np
import tensorflow as tf

from examples.dlrm import model_jce
from examples.dlrm_benchmark import common
from examples.dlrm_benchmark import export_benchmark_model
from xla_saved_model.protos import orchestration_pb2
from xla_saved_model._src.export import jax_model
from xla_saved_model._src.export import main_lib
from xla_saved_model._src.export import serving_config
from xla_saved_model._src.export import tensorflow_model

TF_LOOKUP_FUSED = 'fused'
TF_LOOKUP_GATHER = 'gather'
TF_LOOKUPS = (TF_LOOKUP_FUSED, TF_LOOKUP_GATHER)

TF_PREPROCESSOR_NAME = 'tf_preprocessor'
TF_PREPROCESSOR_METHOD = 'preprocessing'
# Sub-directory of the export that holds the TF SavedModel (tf2xsm layout).
TF_SAVED_MODEL_SUBDIR = 'tf_saved_model'


@dataclasses.dataclass(frozen=True)
class _Flags:
  """Command-line flags; only registered when run as a binary."""

  weights_dir: flags.FlagHolder[str | None]
  batched_export_path: flags.FlagHolder[str | None]
  unbatched_export_path: flags.FlagHolder[str | None]
  tf_lookup: flags.FlagHolder[str]
  platforms: flags.FlagHolder[list[str]]
  batch_component: flags.FlagHolder[str]
  batch_timeout_micros: flags.FlagHolder[int]
  num_batch_threads: flags.FlagHolder[int]
  max_enqueued_batches: flags.FlagHolder[int]


def _define_flags() -> _Flags:
  """Registers flags (lazily, so tests can import this module)."""
  return _Flags(
      weights_dir=flags.DEFINE_string(
          'weights_dir',
          None,
          'Directory with tables.npz / params.npz / config.json written by '
          'export_benchmark_model --baseline_weights_dir.',
      ),
      batched_export_path=flags.DEFINE_string(
          'batched_export_path',
          None,
          'Where to write the XSE model (TF preprocessor) exported with '
          'server-side batching enabled.',
      ),
      unbatched_export_path=flags.DEFINE_string(
          'unbatched_export_path',
          None,
          'Where to write the XSE model (TF preprocessor) exported with '
          'NO_BATCHING.',
      ),
      tf_lookup=flags.DEFINE_enum(
          'tf_lookup',
          TF_LOOKUP_FUSED,
          list(TF_LOOKUPS),
          'TF embedding lookup implementation: "fused" = SparseSegmentSum '
          '(embedding_lookup_sparse fast path), "gather" = ResourceGather + '
          'reduce_sum (TPUEmbeddingForServing / dense embedding_lookup).',
      ),
      platforms=flags.DEFINE_list(
          'platforms',
          ['cpu', 'cuda', 'tpu'],
          'jax.export platforms to lower the accelerator model for.',
      ),
      # Same names and defaults as export_benchmark_model so run_benchmark.sh
      # can pass one flag list to both exporters.
      batch_component=flags.DEFINE_enum(
          'batch_component',
          'PRE_PROCESSOR_AND_MODEL_FUNCTION',
          ['PRE_PROCESSOR_AND_MODEL_FUNCTION', 'MODEL_FUNCTION'],
          'Which pipeline components the batch scheduler wraps for the '
          'batched export (see export_benchmark_model).',
      ),
      batch_timeout_micros=flags.DEFINE_integer(
          'batch_timeout_micros', 1000, 'Max time to wait to fill a batch.'
      ),
      num_batch_threads=flags.DEFINE_integer(
          'num_batch_threads', 2, 'Number of batch scheduler threads.'
      ),
      max_enqueued_batches=flags.DEFINE_integer(
          'max_enqueued_batches', 64, 'Queue depth before requests fail fast.'
      ),
  )


# --------------------------------------------------------------------------- #
# TF preprocessor
# --------------------------------------------------------------------------- #


class TfEmbeddingTables(tf.Module):
  """The 26 embedding tables as non-trainable `tf.Variable`s.

  A `tf.Module` so that `tf.saved_model.save` tracks the variables and the
  server restores them into the TFRT resource manager at load time (and so
  that TFRT can hoist the `ReadVariableOp`s out of the per-request graph).
  """

  def __init__(
      self, tables: Sequence[np.ndarray], name: str = 'tf_embedding_tables'
  ):
    super().__init__(name=name)
    with self.name_scope:
      self.tables = [
          tf.Variable(t, trainable=False, dtype=tf.float32, name=f'table_{i}')
          for i, t in enumerate(tables)
      ]


def tf_embedding_lookup(
    tables: Sequence[tf.Variable],
    features: Mapping[str, tf.Tensor],
    impl: str,
) -> dict[str, tf.Tensor]:
  """SUM-pooled lookup of every feature in TF: `{'i': [B, D]}`.

  Numerically equivalent to `common.embedding_lookup_numpy` and to the JCE
  `CpuEmbeddingModule(dense_lookup=True, combine_mode='SUM')`.

  Args:
    tables: Embedding tables in feature order (`TfEmbeddingTables.tables`).
    features: Request features; int32 `[B, V_i]` ids for table i under key
      `str(i)` (other keys are ignored).
    impl: `TF_LOOKUP_FUSED` or `TF_LOOKUP_GATHER` (see module docstring).

  Returns:
    Pooled embeddings keyed like the inputs.
  """
  if impl not in TF_LOOKUPS:
    raise ValueError(
        f'Unknown TF lookup {impl!r}; expected one of {TF_LOOKUPS}'
    )
  out: dict[str, tf.Tensor] = {}
  # `segment_ids` only depend on (B, V); build each distinct valency once per
  # trace so the graph has a handful of Range/Repeat ops instead of 26.
  segment_ids_by_valency: dict[int, tf.Tensor] = {}
  batch = None
  for i, table in enumerate(tables):
    ids = features[str(i)]
    valency = int(ids.shape[1])
    if impl == TF_LOOKUP_GATHER:
      # ResourceGather -> [B, V, D] -> Sum over V.
      out[str(i)] = tf.reduce_sum(tf.gather(table, ids), axis=1)
      continue
    # Fused path. Read the variable without a copy (what
    # embedding_lookup_sparse does for resource variables); under TFRT the
    # read is invariant and hoisted out of the request graph.
    read = getattr(table, 'read_value_no_copy', None)
    params = read() if read is not None else tf.convert_to_tensor(table)
    if valency == 1:
      out[str(i)] = tf.gather(params, ids[:, 0])
      continue
    if batch is None:
      batch = tf.shape(ids)[0]
    if valency not in segment_ids_by_valency:
      segment_ids_by_valency[valency] = tf.repeat(tf.range(batch), valency)
    out[str(i)] = tf.sparse.segment_sum(
        params,
        tf.reshape(ids, [-1]),
        segment_ids_by_valency[valency],
        num_segments=batch,
    )
  return out


def make_tf_preprocessor(
    module: TfEmbeddingTables, config: common.ModelConfig, impl: str
):
  """Returns the `tf.function` the server runs as `ROLE_PREPROCESSOR`.

  Output: the 26 pooled embeddings under their feature names plus the dense
  features passed through -- the same 27 tensors the JCE preprocessor emits,
  as a flat dict (`tf.nest` and JAX flatten dict keys in the same sorted
  order, which is how they reach the `JaxModel`).

  Args:
    module: Holder of the table variables.
    config: Model config (number of tables).
    impl: TF lookup implementation.

  Returns:
    A `tf.function` of one dict argument.
  """
  num_tables = config.num_tables

  @tf.function
  def preprocessing(inputs: Mapping[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    sparse_inputs = {str(i): inputs[str(i)] for i in range(num_tables)}
    outputs = tf_embedding_lookup(module.tables, sparse_inputs, impl)
    outputs[common.DENSE_FEATURE_NAME] = inputs[common.DENSE_FEATURE_NAME]
    return outputs

  return preprocessing


def tf_input_signature(config: common.ModelConfig) -> dict[str, tf.TensorSpec]:
  """`ServingConfig.input_signature`: dynamic batch, fixed valency per feature."""
  signature = {
      str(i): tf.TensorSpec(shape=(None, v), dtype=tf.int32)
      for i, v in enumerate(config.valency)
  }
  signature[common.DENSE_FEATURE_NAME] = tf.TensorSpec(
      shape=(None, config.num_dense_features), dtype=tf.float32
  )
  return signature


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def _make_apply_fn(config: common.ModelConfig):
  """DCNv2 apply function over the preprocessor's dict output."""
  model = model_jce.build_model(embedding_size=config.embedding_size)
  num_tables = config.num_tables

  def apply_fn(variables, inputs):
    sparse_embeddings = {str(i): inputs[str(i)] for i in range(num_tables)}
    predictions = model.apply(
        variables, inputs[common.DENSE_FEATURE_NAME], sparse_embeddings
    )
    return {common.OUTPUT_NAME: predictions}

  return apply_fn


def _export_xse_tf(
    export_path: str,
    config: common.ModelConfig,
    module: TfEmbeddingTables,
    params: Any,
    platforms: Sequence[str],
    tf_lookup: str,
    batch_opts: orchestration_pb2.BatchOptions,
) -> None:
  """Writes one XSE model directory (TF preprocessor + DCNv2 model)."""
  input_signature = tf_input_signature(config)
  feature_names = common.sparse_feature_names(config.num_tables)

  preprocessor = tensorflow_model.TensorflowModel(
      name=TF_PREPROCESSOR_NAME,
      tf_functions={
          TF_PREPROCESSOR_METHOD: make_tf_preprocessor(
              module, config, tf_lookup
          )
      },
      trackable_resources=module,
  )

  polymorphic_shapes = {n: 'b, _' for n in feature_names}
  polymorphic_shapes[common.DENSE_FEATURE_NAME] = 'b, _'
  jm = jax_model.JaxModel(
      apply_fn_bundle=jax_model.ApplyFnBundle(
          apply_fn=_make_apply_fn(config),
          input_polymorphic_shapes=polymorphic_shapes,
          jax_export_kwargs={'platforms': list(platforms)},
          input_polymorphic_shape_symbol_values={'b': list(config.batch_sizes)},
      ),
      params=params,
      name='jax_model',
  )

  t0 = time.perf_counter()
  main_lib.save(
      models=[preprocessor, jm],
      export_path=export_path,
      serving_configs=[
          serving_config.ServingConfig(
              serving_signature=common.DEFAULT_SIGNATURE,
              input_signature=input_signature,
              batch_options=batch_opts,
              orchestration=serving_config.Orchestration(
                  model_calls=[
                      serving_config.ModelCall(
                          model_name=TF_PREPROCESSOR_NAME,
                          method_key=TF_PREPROCESSOR_METHOD,
                          role=orchestration_pb2.ROLE_PREPROCESSOR,
                      ),
                      serving_config.ModelCall(
                          model_name='jax_model',
                          method_key='serving_default',
                          role=orchestration_pb2.ROLE_MODEL,
                      ),
                  ]
              ),
          )
      ],
      warmup_examples=export_benchmark_model.make_warmup_examples(config),
  )
  logging.info(
      'XSE export (TF %s preprocessor) written to %s in %.1fs '
      '(batch_component=%s)',
      tf_lookup,
      export_path,
      time.perf_counter() - t0,
      orchestration_pb2.BatchOptions.BatchComponent.Name(
          batch_opts.batch_component
      ),
  )


def export_tf_model(
    *,
    weights_dir: str,
    batched_export_path: str | None = None,
    unbatched_export_path: str | None = None,
    tf_lookup: str = TF_LOOKUP_FUSED,
    platforms: Sequence[str] = ('cpu', 'cuda', 'tpu'),
    batch_component: str = 'PRE_PROCESSOR_AND_MODEL_FUNCTION',
    batch_timeout_micros: int = 1000,
    num_batch_threads: int = 2,
    max_enqueued_batches: int = 64,
) -> common.ModelConfig:
  """Loads the JCE exporter's weights and writes the TF-preprocessor exports.

  Args:
    weights_dir: `--baseline_weights_dir` of `export_benchmark_model`.
    batched_export_path: Export with server-side batching, or None.
    unbatched_export_path: Export with `NO_BATCHING`, or None.
    tf_lookup: `TF_LOOKUP_FUSED` or `TF_LOOKUP_GATHER`.
    platforms: `jax.export` platforms for the DCNv2 model.
    batch_component: See `export_benchmark_model.batch_options`.
    batch_timeout_micros: See `export_benchmark_model.batch_options`.
    num_batch_threads: See `export_benchmark_model.batch_options`.
    max_enqueued_batches: See `export_benchmark_model.batch_options`.

  Returns:
    The model config read from `weights_dir`.
  """
  if not (batched_export_path or unbatched_export_path):
    raise ValueError('Nothing to export: specify at least one export path.')
  if tf_lookup not in TF_LOOKUPS:
    raise ValueError(
        f'--tf_lookup must be one of {TF_LOOKUPS}, got {tf_lookup}'
    )

  t0 = time.perf_counter()
  config, tables, params = common.load_baseline_weights(weights_dir)
  logging.info(
      'Loaded weights from %s in %.1fs; config: %s',
      weights_dir,
      time.perf_counter() - t0,
      config,
  )

  # One copy of the tables as tf.Variables, shared by both exports.
  t0 = time.perf_counter()
  module = TfEmbeddingTables(tables)
  del tables  # The variables own a copy; drop the numpy one.
  logging.info('TF variables built in %.1fs', time.perf_counter() - t0)

  for path, batched in (
      (batched_export_path, True),
      (unbatched_export_path, False),
  ):
    if not path:
      continue
    _export_xse_tf(
        export_path=path,
        config=config,
        module=module,
        params=params,
        platforms=platforms,
        tf_lookup=tf_lookup,
        batch_opts=export_benchmark_model.batch_options(
            config,
            batched=batched,
            batch_component=batch_component,
            batch_timeout_micros=batch_timeout_micros,
            num_batch_threads=num_batch_threads,
            max_enqueued_batches=max_enqueued_batches,
        ),
    )
    config.save(os.path.join(path, 'assets.extra'))
  return config


def main(argv: Sequence[str], f: _Flags) -> None:
  if len(argv) > 1:
    raise app.UsageError(f'Unexpected arguments: {argv[1:]}')
  if not f.weights_dir.value:
    raise app.UsageError('--weights_dir is required')
  export_tf_model(
      weights_dir=f.weights_dir.value,
      batched_export_path=f.batched_export_path.value,
      unbatched_export_path=f.unbatched_export_path.value,
      tf_lookup=f.tf_lookup.value,
      platforms=f.platforms.value,
      batch_component=f.batch_component.value,
      batch_timeout_micros=f.batch_timeout_micros.value,
      num_batch_threads=f.num_batch_threads.value,
      max_enqueued_batches=f.max_enqueued_batches.value,
  )


if __name__ == '__main__':
  # The exporter must never grab the accelerator (the server will own it, and
  # the ~3 GB of table variables would otherwise be placed on GPU:0 by a
  # CUDA-enabled TF wheel): hide GPUs from TF before the first op, and keep
  # JAX on CPU as export_benchmark_model does.
  try:
    tf.config.set_visible_devices([], 'GPU')
  except (RuntimeError, ValueError):  # devices already initialised / none
    pass
  if not os.environ.get('JAX_PLATFORMS'):
    jax.config.update('jax_platforms', 'cpu')
  app.run(functools.partial(main, f=_define_flags()))
