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

"""Script to run Neptune export for NNX model with JAX preprocessor."""

from absl import app
from absl import flags
from examples.nnx_jax_preprocessor import export

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'export_path',
    '/tmp/neptune_nnx_jax_preprocessor_export',
    'Path to export model.',
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  export.export_model(FLAGS.export_path)


if __name__ == '__main__':
  app.run(main)
