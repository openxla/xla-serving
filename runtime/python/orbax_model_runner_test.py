"""Test orbax model runner python API on CPU."""


import os
from .pyglib import resources
from .testing.pybase import googletest
from .third_party.neptune.runtime.python import orbax_model_runner
from .third_party.neptune.runtime.python import orbax_model_runner_test_base


class OrbaxModelRunnerTest(
    orbax_model_runner_test_base.OrbaxModelRunnerTestBase
):

  def test_ifrt_platform_name_and_device_count(self):
    model_path = os.path.join(
        resources.GetRunfilesDir(),
        '/learning/infra/mira/experimental/orbax_model/interpreter/testdata/linear_model_gen',
    )
    runner = orbax_model_runner.ModelRunner(
        model_name='test_model', model_path=model_path
    )
    self.assertEqual(runner.ifrt_platform_name(), 'cpu')
    self.assertEqual(runner.ifrt_device_count(), 4)


if __name__ == '__main__':
  googletest.main()
