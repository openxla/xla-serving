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

"""Script to run Neptune export for the ViT image classifier."""

from absl import app
from absl import flags
# Importing `examples.vit` sets KERAS_BACKEND=jax before keras is imported.
from examples.vit import export

_EXPORT_PATH = flags.DEFINE_string('export_path', None, 'Path to export model.')
_DTYPE = flags.DEFINE_enum(
    'dtype',
    'fp32',
    ['fp32', 'bf16'],
    'Model compute dtype. "bf16" uses the Keras mixed_bfloat16 policy '
    '(inputs remain fp32, compute in bf16).',
)
_PRESET = flags.DEFINE_string(
    'preset',
    export.DEFAULT_MODEL_PRESET,
    'keras_hub preset name or local path to load into '
    'ViTImageClassifier.from_preset.',
)
flags.mark_flag_as_required('export_path')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  assert _EXPORT_PATH.value is not None
  assert _PRESET.value is not None
  export.export_model(
      _EXPORT_PATH.value, dtype=_DTYPE.value, preset=_PRESET.value
  )


if __name__ == '__main__':
  app.run(main)
