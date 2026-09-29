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

"""pytest configuration for XLA Serving."""

import importlib.util
import os
from typing import Any
from absl import flags

os.environ.setdefault("KERAS_BACKEND", "jax")

# TensorFlow is optional (the `tf` extra) and isn't available on every Python
# version we test with, so skip the tests and examples that require it.
if importlib.util.find_spec("tensorflow") is None:
  collect_ignore = [
      "examples/dlrm",
      "examples/nnx_tf",
      "examples/nnx_tf_ragged",
      "examples/nnx_with_tf_embeddings",
      "xla_saved_model/_src/export/main_lib_orchestration_test.py",
      "xla_saved_model/_src/processors/tf_processor_test.py",
      "xla_saved_model/_src/tf2xsm",
  ]


def pytest_configure(config: Any) -> None:
  del config
  flags.FLAGS.mark_as_parsed()


try:
  flags.FLAGS.mark_as_parsed()
except Exception:  # pylint: disable=broad-except
  pass
