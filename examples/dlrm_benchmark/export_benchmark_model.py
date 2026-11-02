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

"""Exports the DLRM-DCNv2 benchmark model for XSE *and* the Python baseline.

One invocation generates a single set of seeded weights (26 embedding tables +
DCNv2 parameters) and writes any subset of:

* `--batched_export_path`:   XSE model with server-side `BatchOptions`.
* `--unbatched_export_path`: identical XSE model with `NO_BATCHING`.
* `--baseline_weights_dir`:  `tables.npz` / `params.npz` / `config.json`
                             consumed by `python_server.py`.

Because all three come from the same in-memory arrays, the arms are guaranteed
to serve the same function (verified again at run time by `parity_check.py`).

The XSE export uses the JCE (JAX CPU Embedding) `CompositeProcessor`
preprocessor: the 26 sparse-id lookups + SUM pooling run on the host in C++ and
the dense DCNv2 runs on the accelerator. No TensorFlow is involved.
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
import jax.numpy as jnp
import numpy as np

from xla_serving_data.cpu_embedding.python import embedding_utils
from xla_serving_data.kernels.cpu_embedding import cpu_embedding_module
from examples.dlrm import model_jce
from examples.dlrm_benchmark import common
from xla_saved_model.protos import orchestration_pb2
from xla_saved_model._src.export import jax_model
from xla_saved_model._src.export import main_lib
from xla_saved_model._src.export import serving_config


DEFAULT_JCE_MAX_CPU_SLOTS = 8


@dataclasses.dataclass(frozen=True)
class _Flags:
  """Command-line flags; only registered when run as a binary."""

  batched_export_path: flags.FlagHolder[str | None]
  unbatched_export_path: flags.FlagHolder[str | None]
  baseline_weights_dir: flags.FlagHolder[str | None]
  embedding_size: flags.FlagHolder[int]
  max_vocab_size: flags.FlagHolder[int]
  platforms: flags.FlagHolder[list[str]]
  batch_sizes: flags.FlagHolder[list[str]]
  batch_component: flags.FlagHolder[str]
  batch_timeout_micros: flags.FlagHolder[int]
  num_batch_threads: flags.FlagHolder[int]
  max_enqueued_batches: flags.FlagHolder[int]
  jce_max_cpu_slots: flags.FlagHolder[int]
  seed: flags.FlagHolder[int]


def _define_flags() -> _Flags:
  """Registers flags.

  Defined lazily (not at import time) so that tests and other scripts that
  import this module can define flags with the same names.

  Returns:
    Holders for all flags of this binary.
  """
  return _Flags(
      batched_export_path=flags.DEFINE_string(
          'batched_export_path',
          None,
          'Where to write the XSE model exported with server-side batching '
          'enabled.',
      ),
      unbatched_export_path=flags.DEFINE_string(
          'unbatched_export_path',
          None,
          'Where to write the XSE model exported with NO_BATCHING.',
      ),
      baseline_weights_dir=flags.DEFINE_string(
          'baseline_weights_dir',
          None,
          'Where to write tables.npz/params.npz/config.json for '
          'python_server.py.',
      ),
      embedding_size=flags.DEFINE_integer(
          'embedding_size', 128, 'Embedding dimension of all 26 tables.'
      ),
      max_vocab_size=flags.DEFINE_integer(
          'max_vocab_size',
          0,
          'Cap every vocabulary at this many rows (0 = full Criteo-like sizes, '
          '~6.2M rows total). Use a small value for smoke tests.',
      ),
      platforms=flags.DEFINE_list(
          'platforms',
          ['cpu', 'cuda', 'tpu'],
          'jax.export platforms to lower the accelerator model for.',
      ),
      batch_sizes=flags.DEFINE_list(
          'batch_sizes',
          [str(b) for b in common.DEFAULT_BATCH_SIZES],
          'Concrete batch sizes to specialise the model for. With batching '
          'enabled these are also the `allowed_batch_sizes` the scheduler '
          'pads to.',
      ),
      batch_component=flags.DEFINE_enum(
          'batch_component',
          'PRE_PROCESSOR_AND_MODEL_FUNCTION',
          ['PRE_PROCESSOR_AND_MODEL_FUNCTION', 'MODEL_FUNCTION'],
          'Which pipeline components the batch scheduler wraps for the '
          'batched export. PRE_PROCESSOR_AND_MODEL_FUNCTION batches the CPU '
          'embedding lookup too (one C++ lookup per batch); MODEL_FUNCTION '
          'runs the lookup per request.',
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
      jce_max_cpu_slots=flags.DEFINE_integer(
          'jce_max_cpu_slots',
          DEFAULT_JCE_MAX_CPU_SLOTS,
          'Size of the C++ JCE lookup thread pool (`max_cpu_slots`) baked into '
          'the exported bundle. Match `--num_threads` so XSE and Python use '
          'the same lookup thread budget.',
      ),
      seed=flags.DEFINE_integer('seed', 0, 'Seed for weights.'),
  )


def _build_weights(
    config: common.ModelConfig,
) -> tuple[dict[str, np.ndarray], Any]:
  """Generates seeded embedding tables and DCNv2 params."""
  rng = np.random.default_rng(config.seed)
  tables = {}
  t0 = time.perf_counter()
  for i, vocab in enumerate(config.vocab_sizes):
    tables[str(i)] = rng.uniform(
        -0.05, 0.05, size=(vocab, config.embedding_size)
    ).astype(np.float32)
  total_rows = sum(config.vocab_sizes)
  logging.info(
      'Generated %d tables, %d rows x %d (%.2f GiB) in %.1fs',
      len(tables),
      total_rows,
      config.embedding_size,
      total_rows * config.embedding_size * 4 / 2**30,
      time.perf_counter() - t0,
  )

  model = model_jce.build_model(embedding_size=config.embedding_size)
  init_batch = 2
  dense = jnp.zeros((init_batch, config.num_dense_features), jnp.float32)
  sparse = {
      str(i): jnp.zeros((init_batch, config.embedding_size), jnp.float32)
      for i in range(config.num_tables)
  }
  params = model.init(jax.random.PRNGKey(config.seed), dense, sparse)
  # Materialise on host as plain nested dicts of numpy arrays so that the
  # exact same bytes go to the XSE checkpoint and to params.npz.
  params = jax.tree.map(np.asarray, params)
  params = common.unflatten_params(common.flatten_params(params))
  return tables, params


def _make_apply_fn(config: common.ModelConfig):
  """Returns the DCNv2 apply function in the preprocessor's output order."""
  model = model_jce.build_model(embedding_size=config.embedding_size)
  num_tables = config.num_tables

  def apply_fn(variables, inputs):
    # The preprocessor emits 27 positional tensors: 26 pooled embeddings
    # (feature order) followed by the dense passthrough.
    sparse_embeddings = {str(i): inputs[i] for i in range(num_tables)}
    dense_features = inputs[num_tables]
    predictions = model.apply(variables, dense_features, sparse_embeddings)
    return {common.OUTPUT_NAME: predictions}

  return apply_fn


def _batch_options(
    config: common.ModelConfig,
    batched: bool,
    batch_component: str,
    batch_timeout_micros: int,
    num_batch_threads: int,
    max_enqueued_batches: int,
) -> orchestration_pb2.BatchOptions:
  """`BatchOptions` for the batched (PAD_UP to buckets) or NO_BATCHING export."""
  if not batched:
    return orchestration_pb2.BatchOptions(
        batch_component=orchestration_pb2.BatchOptions.NO_BATCHING
    )
  return orchestration_pb2.BatchOptions(
      batch_component=getattr(orchestration_pb2.BatchOptions, batch_component),
      allowed_batch_sizes=list(config.batch_sizes),
      max_batch_size=config.batch_sizes[-1],
      batch_timeout_micros=batch_timeout_micros,
      num_batch_threads=num_batch_threads,
      max_enqueued_batches=max_enqueued_batches,
      batch_padding_policy=orchestration_pb2.BatchOptions.PAD_UP,
  )


def _export_xse(
    export_path: str,
    config: common.ModelConfig,
    tables: Mapping[str, np.ndarray],
    params: Any,
    platforms: Sequence[str],
    batch_options: orchestration_pb2.BatchOptions,
    max_cpu_slots: int = DEFAULT_JCE_MAX_CPU_SLOTS,
) -> None:
  """Writes one XSE model directory (JCE preprocessor + DCNv2 model)."""
  feature_names = common.sparse_feature_names(config.num_tables)
  feature_specs = [{'name': n, 'table_name': n} for n in feature_names]
  combine_modes = {n: 'SUM' for n in feature_names}

  t0 = time.perf_counter()
  jce_resource = cpu_embedding_module.CpuEmbeddingResource(
      table_data=dict(tables),
      combine_modes=combine_modes,
      feature_specs=feature_specs,
      max_cpu_slots=max_cpu_slots,
  )
  logging.info(
      'JCE resource built in %.1fs (max_cpu_slots=%d)',
      time.perf_counter() - t0,
      max_cpu_slots,
  )

  # Dynamic batch dimension; the second dimension is the fixed valency.
  input_signature = {
      n: jax.ShapeDtypeStruct(shape=(-1, config.valency[i]), dtype=jnp.int32)
      for i, n in enumerate(feature_names)
  }
  input_signature[common.DENSE_FEATURE_NAME] = jax.ShapeDtypeStruct(
      shape=(-1, config.num_dense_features), dtype=jnp.float32
  )

  preprocessor = embedding_utils.build_embedding_lookup(
      name='jce_preprocessor',
      input_signature=input_signature,
      feature_specs=feature_specs,
      jce_resource=jce_resource,
      embedding_dim=config.embedding_size,
      dense_lookup=True,
  )

  jm = jax_model.JaxModel(
      apply_fn_bundle=jax_model.ApplyFnBundle(
          apply_fn=_make_apply_fn(config),
          input_polymorphic_shapes=['b, _'] * (config.num_tables + 1),
          jax_export_kwargs={'platforms': list(platforms)},
          input_polymorphic_shape_symbol_values={'b': list(config.batch_sizes)},
      ),
      params=params,
      name='jax_model',
  )

  # One warmup request per allowed batch size so `--warmup_server=true`
  # compiles/caches every shape before the load generator connects.
  rng = np.random.default_rng(config.seed + 1)
  warmup_examples = {
      common.DEFAULT_SIGNATURE: [
          common.generate_features(config, b, rng) for b in config.batch_sizes
      ]
  }

  t0 = time.perf_counter()
  main_lib.save(
      models=[preprocessor, jm],
      export_path=export_path,
      serving_configs=[
          serving_config.ServingConfig(
              serving_signature=common.DEFAULT_SIGNATURE,
              input_signature=input_signature,
              batch_options=batch_options,
              orchestration=serving_config.Orchestration(
                  model_calls=[
                      serving_config.ModelCall(
                          model_name='jce_preprocessor',
                          method_key='serving_default',
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
      warmup_examples=warmup_examples,
  )
  logging.info(
      'XSE export written to %s in %.1fs (batch_component=%s)',
      export_path,
      time.perf_counter() - t0,
      orchestration_pb2.BatchOptions.BatchComponent.Name(
          batch_options.batch_component
      ),
  )
  jce_resource.close()


def export_model(
    *,
    embedding_size: int = 128,
    max_vocab_size: int = 0,
    batch_sizes: Sequence[int] = common.DEFAULT_BATCH_SIZES,
    platforms: Sequence[str] = ('cpu', 'cuda', 'tpu'),
    batched_export_path: str | None = None,
    unbatched_export_path: str | None = None,
    baseline_weights_dir: str | None = None,
    batch_component: str = 'PRE_PROCESSOR_AND_MODEL_FUNCTION',
    batch_timeout_micros: int = 1000,
    num_batch_threads: int = 2,
    max_enqueued_batches: int = 64,
    jce_max_cpu_slots: int = DEFAULT_JCE_MAX_CPU_SLOTS,
    seed: int = 0,
) -> common.ModelConfig:
  """Generates weights once and writes the requested artifacts."""
  if not (batched_export_path or unbatched_export_path or baseline_weights_dir):
    raise ValueError('Nothing to export: specify at least one output path.')
  if (batched_export_path or unbatched_export_path) and jce_max_cpu_slots < 1:
    raise ValueError(
        f'jce_max_cpu_slots must be >= 1, got {jce_max_cpu_slots}'
    )
  if batched_export_path:
    if num_batch_threads < 1:
      raise ValueError(
          f'num_batch_threads must be >= 1, got {num_batch_threads}'
      )
    if max_enqueued_batches < 1:
      raise ValueError(
          f'max_enqueued_batches must be >= 1, got {max_enqueued_batches}'
      )
    if batch_timeout_micros < 0:
      raise ValueError(
          f'batch_timeout_micros must be >= 0, got {batch_timeout_micros}'
      )
  config = common.make_model_config(
      embedding_size=embedding_size,
      max_vocab_size=max_vocab_size,
      batch_sizes=batch_sizes,
      seed=seed,
  )
  logging.info('Model config: %s', config)
  tables, params = _build_weights(config)

  if baseline_weights_dir:
    t0 = time.perf_counter()
    common.save_baseline_weights(baseline_weights_dir, config, tables, params)
    logging.info(
        'Baseline weights written to %s in %.1fs',
        baseline_weights_dir,
        time.perf_counter() - t0,
    )

  for path, batched in (
      (batched_export_path, True),
      (unbatched_export_path, False),
  ):
    if not path:
      continue
    _export_xse(
        export_path=path,
        config=config,
        tables=tables,
        params=params,
        platforms=platforms,
        batch_options=_batch_options(
            config,
            batched=batched,
            batch_component=batch_component,
            batch_timeout_micros=batch_timeout_micros,
            num_batch_threads=num_batch_threads,
            max_enqueued_batches=max_enqueued_batches,
        ),
        max_cpu_slots=jce_max_cpu_slots,
    )
    # Also drop the baseline config next to the export for convenience.
    config.save(os.path.join(path, 'assets.extra'))
  return config


def main(argv: Sequence[str], f: _Flags) -> None:
  if len(argv) > 1:
    raise app.UsageError(f'Unexpected arguments: {argv[1:]}')
  export_model(
      embedding_size=f.embedding_size.value,
      max_vocab_size=f.max_vocab_size.value,
      batch_sizes=[int(b) for b in f.batch_sizes.value],
      platforms=f.platforms.value,
      batched_export_path=f.batched_export_path.value,
      unbatched_export_path=f.unbatched_export_path.value,
      baseline_weights_dir=f.baseline_weights_dir.value,
      batch_component=f.batch_component.value,
      batch_timeout_micros=f.batch_timeout_micros.value,
      num_batch_threads=f.num_batch_threads.value,
      max_enqueued_batches=f.max_enqueued_batches.value,
      jce_max_cpu_slots=f.jce_max_cpu_slots.value,
      seed=f.seed.value,
  )


if __name__ == '__main__':
  # Make sure the exporter itself never grabs the accelerator: all exported
  # artifacts are platform-independent (`jax.export` lowers for `--platforms`
  # without the hardware) and the server will own the device. JAX reads
  # `JAX_PLATFORMS` from the environment at import time, so it has to be set
  # through the config here; an explicit environment value still wins.
  if not os.environ.get('JAX_PLATFORMS'):
    jax.config.update('jax_platforms', 'cpu')
  app.run(functools.partial(main, f=_define_flags()))
