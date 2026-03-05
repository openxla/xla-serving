"""Tests for orbax_model_runner.cc CUDA backend."""

import os
import numpy as np
from google3.pyglib import resources
from google3.testing.pybase import googletest
from google3.third_party.neptune.runtime.python import orbax_model_runner

_TEST_DATA_DIR = (
    'google3/learning/infra/mira/experimental/orbax_model/interpreter/testdata'
)


class OrbaxModelRunnerCudaTest(googletest.TestCase):
  """Test suite for CUDA model runner."""

  def test_load_and_run_stablehlo_sharded_succeeds(self):
    """Tests loading a model and running a StableHlo function on CUDA."""
    model_path = os.path.join(
        resources.GetRunfilesDir(),
        _TEST_DATA_DIR,
        'linear_model_2x1_sharded_gen',
    )
    runner = orbax_model_runner.ModelRunner(model_path=model_path)

    # Check platform and device count
    self.assertEqual(runner.ifrt_platform_name(), 'cuda')
    self.assertEqual(runner.ifrt_device_count(), 2)

    signature_names = ('linear', 'serving_default')
    self.assertSameElements(runner.signature_names(), signature_names)

    signature = runner.signature(signature_names[0])

    batch_size = 4
    input_dim = 2
    inputs = np.arange(batch_size * input_dim, dtype=np.float32).reshape(
        (batch_size, input_dim)
    )

    outputs = signature.run(inputs)
    self.assertLen(outputs, 1)

    expected_output = np.array(
        [[2, 3], [6, 7], [10, 11], [14, 15]],
        dtype=np.float32,
    )
    self.assertTrue(np.array_equal(np.array(outputs[0]), expected_output))


if __name__ == '__main__':
  googletest.main()
