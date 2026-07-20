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

"""Script to run Neptune export for DLRM model."""

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


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  export.export_model(
      _EXPORT_PATH.value,
      use_capped_vocab=_USE_CAPPED_VOCAB.value,
      embedding_size=_EMBEDDING_SIZE.value,
      global_batch_size=_GLOBAL_BATCH_SIZE.value,
      num_dense_features=_NUM_DENSE_FEATURES.value,
  )


if __name__ == '__main__':
  app.run(main)
