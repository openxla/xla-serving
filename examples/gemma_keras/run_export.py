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

"""Script to run XLA SavedModel export for Gemma3."""

from absl import app
from absl import flags
# Importing `examples.gemma_keras` sets KERAS_BACKEND=jax before keras is imported.
from examples.gemma_keras import export

_EXPORT_PATH = flags.DEFINE_string('export_path', None, 'Path to export model.')
_DTYPE = flags.DEFINE_enum(
    'dtype',
    'fp32',
    ['fp32', 'bf16'],
    'Model compute dtype. "bf16" uses the Keras mixed_bfloat16 policy.',
)
_PRESET = flags.DEFINE_string(
    'preset',
    export.DEFAULT_MODEL_PRESET,
    'keras_hub preset name or local path to load into '
    'Gemma3CausalLM.from_preset.',
)
_MAX_SEQ_LEN = flags.DEFINE_integer(
    'max_seq_len',
    export.DEFAULT_MAX_SEQ_LEN,
    'Maximum (padded) prompt length the exported model accepts for prefill.',
)
_MAX_NEW_TOKENS = flags.DEFINE_integer(
    'max_new_tokens',
    export.DEFAULT_MAX_NEW_TOKENS,
    'Upper bound on tokens produced by the in-graph decode loop. The loop '
    'exits early when every batch row has emitted the client-supplied '
    'eos_token_id.',
)
flags.mark_flag_as_required('export_path')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  export.export_model(
      _EXPORT_PATH.value,
      dtype=_DTYPE.value,
      preset=_PRESET.value,
      max_seq_len=_MAX_SEQ_LEN.value,
      max_new_tokens=_MAX_NEW_TOKENS.value,
  )


if __name__ == '__main__':
  app.run(main)
