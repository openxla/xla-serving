# Copyright 2026 Google LLC
# Copyright 2026 Arm Ltd. and affiliates
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

"""Script to run XLA SavedModel export for DLRM model."""

from absl import app
from absl import flags
from examples.dlrm import export_jce as export

_EXPORT_PATH = flags.DEFINE_string('export_path', None, 'Path to export model.')
flags.mark_flag_as_required('export_path')
_USE_CAPPED_VOCAB = flags.DEFINE_bool(
    'use_capped_vocab', False, 'Cap vocab sizes to 100 for fast tests.'
)
_GLOBAL_BATCH_SIZE = flags.DEFINE_integer(
    'global_batch_size', 2, 'Global batch size of the model.'
)
_NUM_DENSE_FEATURES = flags.DEFINE_integer(
    'num_dense_features', 13, 'Number of dense features.'
)
_EMBEDDING_SIZE = flags.DEFINE_integer('embedding_size', 128, 'Embedding size.')
_VOCAB_SIZE_CAP = flags.DEFINE_integer(
    'vocab_size_cap', None, 'Optional per-table vocabulary cap.', lower_bound=1
)
_SEED = flags.DEFINE_integer('seed', 0, 'Seed for synthetic model weights and inputs.')
_BENCHMARK_INPUTS = flags.DEFINE_string(
    'benchmark_inputs', None, 'Optional new .npz file containing inputs and answers.'
)
_NUM_INPUT_BATCHES = flags.DEFINE_integer(
    'num_input_batches', 8, 'Number of distinct batches saved with --benchmark_inputs.',
    lower_bound=1,
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if _EXPORT_PATH.value is None:
    raise app.UsageError('Flag --export_path must be specified.')
  export.export_model(
      _EXPORT_PATH.value,
      use_capped_vocab=_USE_CAPPED_VOCAB.value or False,
      embedding_size=_EMBEDDING_SIZE.value or 128,
      global_batch_size=_GLOBAL_BATCH_SIZE.value or 2,
      num_dense_features=_NUM_DENSE_FEATURES.value or 13,
      vocab_size_cap=_VOCAB_SIZE_CAP.value,
      seed=_SEED.value,
      benchmark_inputs=_BENCHMARK_INPUTS.value,
      num_input_batches=_NUM_INPUT_BATCHES.value,
  )


if __name__ == '__main__':
  app.run(main)
