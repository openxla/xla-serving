"""Base class for orbax model runner python API tests."""

import os
from typing import Final

import numpy as np

from google3.pyglib import resources
from google3.testing.pybase import googletest
from google3.testing.pybase import parameterized
from google3.third_party.neptune.python import orbax_model_runner

_TEST_DATA_DIR: Final[str] = (
    'google3/learning/infra/mira/experimental/orbax_model/interpreter/testdata'
)


# A convenience function to get a signature from a given model. The ModelRunner
# is out of scope after this function returns, so it helps with testing of the
# `ModelRunner`'s lifetime management.
def get_signature(
    model_path: str, signature_name: str
) -> orbax_model_runner.Signature:
  runner = orbax_model_runner.ModelRunner(model_path=model_path)
  return runner.signature(signature_name)


class OrbaxModelRunnerTestBase(parameterized.TestCase):
  """Base class for Orbax model runner tests."""

  def test_load_and_run_stablehlo_succeeds(self):
    """Tests loading a model and running a StableHlo function.

    The model implements a simple linear transformation: y = x * w + b
    weights are
      Array([[1., 1.],
            [1., 1.]], dtype=float32)
    bias are
      Array([1., 2.], dtype=float32)
    """
    model_path = os.path.join(
        resources.GetRunfilesDir(),
        _TEST_DATA_DIR,
        'linear_model_gen',
    )

    runner = orbax_model_runner.ModelRunner(model_path=model_path)
    signature_names = ('linear', 'serving_default')
    self.assertSameElements(runner.signature_names(), signature_names)
    # The two signature names are aliases of each other. So we can use any one
    # of them to get the signature.
    signature = runner.signature(signature_names[0])

    def _create_inputs():
      batch_size = 4
      input_dim = 2
      return np.arange(batch_size * input_dim, dtype=np.float32).reshape(
          (batch_size, input_dim)
      )

    inputs = _create_inputs()
    outputs = signature.run(inputs)
    self.assertLen(outputs, 1)

    expected_output = np.array(
        [[2, 3], [6, 7], [10, 11], [14, 15]],
        dtype=np.float32,
    )
    self.assertTrue(np.array_equal(np.array(outputs[0]), expected_output))

  def test_stablehlo_with_tf_processing_succeeds(self):
    """Tests a model with simple orchestration of JAX and TF functions.

    The JAX function is the same as in the test case
    `test_load_and_run_stablehlo_succeeds`.
    The TF pre-processor function does not change the shape of its input. It is
    a simple transformation on the input: y = x + 1.
    The TF post-processor function does not change the shape of its input. It is
    a simple transformation on the input: y = x * 2 + 1.
    """

    model_path = os.path.join(
        resources.GetRunfilesDir(),
        _TEST_DATA_DIR,
        'simple_model_with_tf_processing_gen',
    )

    signature = get_signature(model_path, 'serving_default')

    def _create_inputs():
      batch_size = 4
      input_dim = 2
      return np.arange(batch_size * input_dim, dtype=np.float32).reshape(
          (batch_size, input_dim)
      )

    inputs = _create_inputs()
    outputs = signature.run(np.array(inputs))
    self.assertLen(outputs, 1)

    expected_output = np.array(
        [[7, 7], [15, 15], [23, 23], [31, 31]],
        dtype=np.float32,
    )
    self.assertTrue(np.array_equal(np.array(outputs[0]), expected_output))

  def test_load_and_run_with_dict_succeeds(self):
    """Tests loading a StableHLO model and running it with dict input/output.

    The model implements a simple function:
    y1 = first_digit * 10000 + second_digit * 1000 + third_digit * 100 +
        fourth_digit * 10 + fifth_digit
    y2 = first_digit * 0.1 + second_digit * 0.01 + third_digit * 0.001 +
        fourth_digit * 0.0001 + fifth_digit * 0.00001

    Both the input and output are dicts. See the model exporting code for more
    details.
    """
    model_path = os.path.join(
        resources.GetRunfilesDir(),
        _TEST_DATA_DIR,
        'make_number_from_digits_dict_gen',
    )
    signature = get_signature(model_path, 'make_number_from_digits')
    # By default, the first_digit is [1, 2], and the second_digit is [3, 4].
    outputs = signature.run_with_dict_return(
        third_digit=np.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.int32),
        fourth_digit=np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int32),
        fifth_digit=np.array([[2, 3], [4, 5], [6, 7], [8, 9]], dtype=np.int32),
    )
    expected_y1 = np.array(
        [[13012, 24123], [13234, 24345], [13456, 24567], [13678, 24789]],
        dtype=np.int32,
    )
    expected_y2 = np.array(
        [
            [0.13012, 0.24123],
            [0.13234, 0.24345],
            [0.13456, 0.24567],
            [0.13678, 0.24789],
        ],
        dtype=np.float32,
    )
    self.assertCountEqual(outputs.keys(), ('y1', 'y2'))
    np.testing.assert_array_equal(outputs['y1'], expected_y1)
    np.testing.assert_allclose(outputs['y2'], expected_y2, rtol=1e-6)

  @parameterized.named_parameters(
      ('bytes', np.bytes_),
      ('str', np.str_),
      ('object', np.object_),
      ('StringDType', np.dtypes.StringDType()),  # type: ignore
  )
  def test_string_io_succeeds(self, string_dtype: np.dtype):
    """Tests a model that has string input and output.

    The model converts a string to a float, multiplies it by a factor (2.0), and
    then converts it back to a string.

    Args:
      string_dtype: The dtype of the string input.
    """
    model_path = os.path.join(
        resources.GetRunfilesDir(),
        _TEST_DATA_DIR,
        'string_to_number_round_trip_batch_model_func_gen',
    )
    signature = get_signature(model_path, 'multiply_by_factor')

    inputs = np.array(['1.0', '2.0', '3.0', '4.0'], dtype=string_dtype)
    outputs = signature.run_with_dict_return(inputs)
    self.assertCountEqual(outputs.keys(), ('output',))
    expected_output = np.array(['2', '4', '6', '8'], dtype=np.bytes_)
    self.assertTrue(np.array_equal(outputs['output'], expected_output))

  @parameterized.named_parameters(
      ('bytes', np.bytes_),
      ('str', np.str_),
      ('object', np.object_),
      ('StringDType', np.dtypes.StringDType()),  # type: ignore
  )
  def test_string_io_and_host_buffer_conversion_succeeds(
      self, string_dtype: np.dtype
  ):
    """Tests a model exported with WHOLE_PIPELINE batching and has string I/O.

    The model converts a string to a float, multiplies it by a factor (2.0), and
    then converts it back to a string. The batch component of the model is
    WHOLE_PIPELINE which requires host buffer conversion for string input and
    output.

    Args:
      string_dtype: The dtype of the string input.
    """
    model_path = os.path.join(
        resources.GetRunfilesDir(),
        _TEST_DATA_DIR,
        'string_to_number_round_trip_batch_whole_pipeline_gen',
    )

    signature = get_signature(model_path, 'multiply_by_factor')

    inputs = np.array(['1.0', '2.0', '3.0', '4.0'], dtype=string_dtype)
    outputs = signature.run(inputs)
    expected_output = np.array(['2', '4', '6', '8'], dtype=np.bytes_)
    self.assertTrue(np.array_equal(outputs[0], expected_output))

  @parameterized.named_parameters(
      ('no_batching', 'string_to_number_round_trip_no_batching_gen'),
      ('batch_model_func', 'string_to_number_round_trip_batch_model_func_gen'),
      (
          'batch_whole_pipeline',
          'string_to_number_round_trip_batch_whole_pipeline_gen',
      ),
      (
          'batch_pre_processor_and_model_func',
          'string_to_number_round_trip_batch_pre_processor_and_model_func_gen',
      ),
      (
          'batch_model_func_and_post_processor',
          'string_to_number_round_trip_batch_model_func_and_post_processor_gen',
      ),
  )
  def test_batch_components_with_string_io_succeeds(
      self, batch_model_path: str
  ):
    """Tests a model with different batch components and string I/O.

    The model converts a string to a float, multiplies it by a factor (2.0), and
    then converts it back to a string. The model was exported with different
    batch components and the expected output should be the same.

    Args:
      batch_model_path: The path to the model with different batch components.
    """
    model_path = os.path.join(
        resources.GetRunfilesDir(),
        _TEST_DATA_DIR,
        batch_model_path,
    )
    signature = get_signature(model_path, 'multiply_by_factor')

    inputs = np.array(['1.0', '2.0', '3.0', '4.0'], dtype=np.bytes_)
    outputs = signature.run(inputs)
    expected_output = np.array(['2', '4', '6', '8'], dtype=np.bytes_)
    self.assertTrue(np.array_equal(outputs[0], expected_output))

  def test_multi_signatures_succeeds(self):
    """Tests a model with multiple signatures.

    This test loads a model with four signatures:
    - plus_sig: y = 1 + x
    - plus2_sig: y = 2 + x
    - minus_sig: y = 1 - x
    - minus2_sig: y = 2 - x
    """
    model_path = os.path.join(
        resources.GetRunfilesDir(),
        _TEST_DATA_DIR,
        'multiple_apply_fn_sharing_ckpt_gen',
    )
    runner = orbax_model_runner.ModelRunner(model_path=model_path)
    self.assertSameElements(
        runner.signature_names(),
        ('plus_sig', 'plus2_sig', 'minus_sig', 'minus2_sig'),
    )

    shape = (4, 2)
    inputs = np.arange(8, dtype=np.float32).reshape(shape)

    expected_output_range = {
        'plus_sig': (1, 9, 1),
        'plus2_sig': (2, 10, 1),
        'minus_sig': (1, -7, -1),
        'minus2_sig': (2, -6, -1),
    }

    for signature_name, (start, stop, step) in expected_output_range.items():
      signature = runner.signature(signature_name)
      expected_output = np.arange(start, stop, step, dtype=np.float32).reshape(
          shape
      )
      # Run with flat input/output.
      outputs = signature.run(inputs)
      self.assertLen(outputs, 1)
      self.assertTrue(np.array_equal(outputs[0], expected_output))
      # Run with dict input/output.
      outputs = signature.run_with_dict_return(x=inputs)
      self.assertCountEqual(outputs.keys(), ('y',))
      self.assertTrue(np.array_equal(outputs['y'], expected_output))

  def test_empty_path_raises_error(self):
    """Tests that an empty model path raises an error."""
    with self.assertRaisesRegex(ValueError, 'Failed to initialize ModelRunner'):
      orbax_model_runner.ModelRunner(model_path='')

  def test_wrong_signature_name_raises_error(self):
    """Tests that an invalid signature name raises an error."""
    model_path = os.path.join(
        resources.GetRunfilesDir(),
        _TEST_DATA_DIR,
        'linear_model_gen',
    )
    runner = orbax_model_runner.ModelRunner(model_path=model_path)
    with self.assertRaisesRegex(
        ValueError,
        'No signature named "non_existent" found in the model',
    ):
      runner.signature('non_existent')


if __name__ == '__main__':
  googletest.main()
