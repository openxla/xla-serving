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

"""Test model runner python API on CPU."""

import os
from .pyglib import resources
from .testing.pybase import googletest
from .third_party.neptune.runtime.python import model_runner
from .third_party.neptune.runtime.python import model_runner_test_base


class ModelRunnerTest(model_runner_test_base.ModelRunnerTestBase):

  def test_ifrt_platform_name_and_device_count(self):
    model_path = os.path.join(
        resources.GetRunfilesDir(),
        '/learning/infra/mira/experimental/orbax_model/interpreter/testdata/linear_model_gen',
    )
    runner = model_runner.ModelRunner(
        model_name='test_model', model_path=model_path
    )
    self.assertEqual(runner.ifrt_platform_name(), 'cpu')
    self.assertEqual(runner.ifrt_device_count(), 4)


if __name__ == '__main__':
  googletest.main()
