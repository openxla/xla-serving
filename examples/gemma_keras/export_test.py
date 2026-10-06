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

"""Tests for XLA SavedModel export of Gemma3 with server-side generation.

Exports with a tiny budget and, if xla_saved_model_runner is importable, checks
the generated_ids name, shape and dtype. Opt-in: skips unless --model_preset
resolves (Gemma3 4B is too large for default CI).
"""

import os

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
# Sets KERAS_BACKEND=jax. `export` is imported in the test, not here: importing
# Keras 3 at collection breaks TF_USE_LEGACY_KERAS for other examples (nnx_tf).
from examples import gemma_keras

_MODEL_PRESET = flags.DEFINE_string(
    'model_preset',
    gemma_keras.DEFAULT_MODEL_PRESET,
    'keras_hub preset name or local path for Gemma3CausalLM. Defaults to '
    f'{gemma_keras.DEFAULT_MODEL_PRESET!r}; the test skips if the preset is not '
    'resolvable (e.g. not downloaded in a CI environment).',
)

# Tiny seq/new-tokens budget so the test stays fast.
_TEST_MAX_SEQ_LEN = 16
_TEST_MAX_NEW_TOKENS = 4


def _preset_available(preset: str) -> bool:
  """Returns True if `preset` resolves.

  Probes with the few-MB tokenizer: from_preset raises ValueError for both a
  missing preset and a broken model, so its errors can't tell skip from fail.
  """
  # Not at module scope: KERAS_BACKEND must be set before keras is imported.
  import keras_hub  # pylint: disable=g-import-not-at-top

  try:
    keras_hub.models.Gemma3Tokenizer.from_preset(preset)
  except Exception as e:  # pylint: disable=broad-except
    logging.warning('Preset %r is not resolvable (%s: %s).', preset,
                    type(e).__name__, e)
    return False
  return True


def _model_runner(model_path: str):
  try:
    from xla_saved_model_runner import model_runner  # pylint: disable=g-import-not-at-top

    return model_runner.ModelRunner(model_path=model_path)
  except ImportError:
    return None


class ExportTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('fp32', 'fp32'),
      ('bf16', 'bf16'),
  )
  def test_export_model(self, dtype):
    preset = _MODEL_PRESET.value
    if not _preset_available(preset):
      self.skipTest(f'Preset {preset!r} not resolvable; this test is opt-in.')
    from examples.gemma_keras import export  # pylint: disable=g-import-not-at-top

    export_path = self.create_tempdir().full_path
    try:
      export.export_model(
          export_path,
          dtype=dtype,
          preset=preset,
          max_seq_len=_TEST_MAX_SEQ_LEN,
          max_new_tokens=_TEST_MAX_NEW_TOKENS,
      )
    except OSError as e:
      # The preset resolved, so this is weight-fetch I/O flakiness, not a bug.
      self.skipTest(
          f'Could not fetch weights for {preset!r} ({type(e).__name__}: {e}).'
      )

    self.assertTrue(os.path.isdir(os.path.join(export_path, 'checkpoint')))
    self.assertTrue(
        os.path.isfile(os.path.join(export_path, 'orchestration.pb'))
    )

    runner = _model_runner(export_path)
    if runner is None:
      logging.warning(
          'xla_saved_model_runner not importable; skipping the round-trip.'
      )
      return

    prompt_len = 4
    batch_size = 1
    token_ids = np.zeros((batch_size, _TEST_MAX_SEQ_LEN), dtype=np.int32)
    prompt_len_arr = np.full((batch_size,), prompt_len, dtype=np.int32)
    # Out-of-vocab EOS so decode fills all max_new_tokens.
    eos_token_id_arr = np.full((batch_size,), 2_000_000_000, dtype=np.int32)

    out = runner.signature('serving_default').run_with_dict_return(
        token_ids=token_ids,
        prompt_len=prompt_len_arr,
        eos_token_id=eos_token_id_arr,
    )
    self.assertIn('generated_ids', out)
    self.assertEqual(out['generated_ids'].dtype, np.int32)
    self.assertEqual(out['generated_ids'].shape, (1, _TEST_MAX_NEW_TOKENS))


if __name__ == '__main__':
  absltest.main()
