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

"""Tests for Neptune export of the ViT image classifier."""

import os

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
# Importing `examples.vit` sets KERAS_BACKEND=jax before keras is imported.
from examples.vit import export

_SAMPLE_IMAGE_PATH = flags.DEFINE_string(
    'sample_image_path',
    '',
    'Optional path to a real JPEG/PNG image. When set (and the file exists), '
    'the test also runs inference on the real image and logs top-5 '
    'predictions — useful for catching accuracy regressions. Skipped when '
    'unset.',
)


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
    export_path = self.create_tempdir().full_path
    export.export_model(export_path, dtype=dtype)
    self.assertTrue(os.path.exists(export_path))
    self.assertTrue(os.path.isdir(os.path.join(export_path, 'checkpoint')))
    self.assertTrue(
        os.path.isfile(os.path.join(export_path, 'orchestration.pb'))
    )
    runner = _model_runner(export_path)
    if runner is None:
      logging.warning(
          'ModelRunner not found, skipping running the exported model'
      )
      return

    test_image = np.random.randn(1, 224, 224, 3).astype(np.float32)
    outputs = runner.signature('serving_default').run_with_dict_return(
        images=test_image,
    )
    self.assertIn('logits', outputs)
    self.assertEqual(outputs['logits'].shape[0], 1)
    # apply_fn casts logits to fp32 regardless of compute dtype; the C++
    # client relies on this (see client.cc's KIND_F32 check).
    self.assertEqual(outputs['logits'].dtype, np.float32)

    top5 = np.argsort(outputs['logits'][0])[-5:][::-1]
    logging.info('Random image top-5 class indices: %s', top5)
    logging.info('Random image top-5 logits: %s', outputs['logits'][0][top5])

    sample_path = _SAMPLE_IMAGE_PATH.value
    if not sample_path:
      logging.info(
          '--sample_image_path not set, skipping real-image inference'
      )
      return
    if not os.path.exists(sample_path):
      logging.warning(
          'Sample image %s not found, skipping real-image inference',
          sample_path,
      )
      return
    try:
      from PIL import Image  # pylint: disable=g-import-not-at-top
    except ImportError:
      logging.warning('PIL not found, skipping real-image inference')
      return

    img = Image.open(sample_path).convert('RGB').resize((224, 224))
    img_array = (np.array(img, dtype=np.float32) / 127.5 - 1.0)[np.newaxis, ...]
    outputs = runner.signature('serving_default').run_with_dict_return(
        images=img_array,
    )
    self.assertIn('logits', outputs)
    self.assertEqual(outputs['logits'].shape[0], 1)
    self.assertEqual(outputs['logits'].dtype, np.float32)
    top5 = np.argsort(outputs['logits'][0])[-5:][::-1]
    logging.info('Real image top-5 class indices: %s', top5)
    logging.info('Real image top-5 logits: %s', outputs['logits'][0][top5])


if __name__ == '__main__':
  absltest.main()
