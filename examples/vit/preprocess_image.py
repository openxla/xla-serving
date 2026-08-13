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

"""Preprocess an image for ViT and save as raw float32 binary."""

from absl import app
from absl import flags
from absl import logging
import numpy as np
from PIL import Image

_IMAGE_PATH = flags.DEFINE_string(
    'image_path', None, 'Path to input JPEG/PNG image.'
)
_OUTPUT = flags.DEFINE_string(
    'output', 'image.bin', 'Output path for raw float32 binary.'
)
flags.mark_flag_as_required('image_path')


def preprocess(image_path: str, output_path: str):
  img = Image.open(image_path).convert('RGB').resize((224, 224))
  img_array = np.array(img, dtype=np.float32) / 127.5 - 1.0
  img_array.tofile(output_path)
  logging.info('Saved preprocessed image to %s', output_path)
  logging.info(
      '  Shape: (224, 224, 3), dtype: float32, size: %d bytes',
      img_array.nbytes,
  )


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  preprocess(_IMAGE_PATH.value, _OUTPUT.value)


if __name__ == '__main__':
  app.run(main)
