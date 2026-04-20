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

"""Tests for serving_config."""

from absl.testing import absltest
from absl.testing import parameterized
from neptune_model._src.export import serving_config
from neptune_model.protos import orchestration_pb2


class ServingConfigTest(parameterized.TestCase):

  def test_default_serving_config(self):
    config = serving_config.ServingConfig()
    self.assertEqual(config.serving_signature, "serving_default")
    self.assertIsNone(config.method_key)
    self.assertIsNone(config.input_signature)
    self.assertIsNone(config.batch_options)

  @parameterized.named_parameters(
      (
          "none",
          None,
          None,
          None,
          None,
      ),
      (
          "no_batching",
          orchestration_pb2.BatchOptions(
              batch_component=orchestration_pb2.BatchOptions.NO_BATCHING
          ),
          0,
          0,
          0,
      ),
      (
          "unspecified_batching",
          orchestration_pb2.BatchOptions(
              batch_component=orchestration_pb2.BatchOptions.BATCH_COMPONENT_UNSPECIFIED
          ),
          0,
          0,
          0,
      ),
      (
          "valid_main",
          orchestration_pb2.BatchOptions(
              batch_component=orchestration_pb2.BatchOptions.MODEL_FUNCTION,
              max_batch_size=8,
              allowed_batch_sizes=[2, 4, 8],
              max_enqueued_batches=1,
              num_batch_threads=2,
          ),
          8,
          2,
          0,
      ),
      (
          "infer_max_batch_size",
          orchestration_pb2.BatchOptions(
              batch_component=orchestration_pb2.BatchOptions.MODEL_FUNCTION,
              allowed_batch_sizes=[2, 4, 16],
              max_enqueued_batches=1,
          ),
          16,
          1,
          0,
      ),
      (
          "adjust_num_threads_zero_to_one",
          orchestration_pb2.BatchOptions(
              batch_component=orchestration_pb2.BatchOptions.MODEL_FUNCTION,
              max_batch_size=8,
              num_batch_threads=0,
              max_enqueued_batches=1,
          ),
          8,
          1,
          0,
      ),
      (
          "low_priority_valid",
          orchestration_pb2.BatchOptions(
              batch_component=orchestration_pb2.BatchOptions.MODEL_FUNCTION,
              max_batch_size=16,
              max_enqueued_batches=1,
              low_priority_batch_options=orchestration_pb2.BatchOptions.LowPriorityBatchOptions(
                  max_batch_size=4,
                  allowed_batch_sizes=[2, 4],
                  max_enqueued_batches=1,
              ),
          ),
          16,
          1,
          4,
      ),
      (
          "low_priority_infer_max_batch_size",
          orchestration_pb2.BatchOptions(
              batch_component=orchestration_pb2.BatchOptions.MODEL_FUNCTION,
              max_batch_size=16,
              max_enqueued_batches=1,
              low_priority_batch_options=orchestration_pb2.BatchOptions.LowPriorityBatchOptions(
                  allowed_batch_sizes=[2, 4, 12], max_enqueued_batches=1
              ),
          ),
          16,
          1,
          12,
      ),
  )
  def test_batch_options_config_transformation(
      self,
      batch_options,
      expected_max_batch_size,
      expected_num_threads,
      expected_lp_max_batch_size,
  ):
    config = serving_config.ServingConfig(batch_options=batch_options)
    if batch_options is None:
      self.assertIsNone(config.batch_options)
      return

    self.assertIsNotNone(config.batch_options)
    assert config.batch_options is not None
    self.assertEqual(
        config.batch_options.max_batch_size, expected_max_batch_size
    )
    self.assertEqual(
        config.batch_options.num_batch_threads, expected_num_threads
    )
    if config.batch_options.HasField("low_priority_batch_options"):
      self.assertEqual(
          config.batch_options.low_priority_batch_options.max_batch_size,
          expected_lp_max_batch_size,
      )

  @parameterized.named_parameters(
      (
          "main_missing_max_batch_size",
          False,
          orchestration_pb2.BatchOptions(
              allowed_batch_sizes=[], max_enqueued_batches=1
          ),
          "max_batch_size must be provided when allowed_batch_sizes is empty",
      ),
      (
          "main_invalid_allowed_batch_sizes_not_positive",
          False,
          orchestration_pb2.BatchOptions(
              max_batch_size=16,
              allowed_batch_sizes=[0, 4, 16],
              max_enqueued_batches=1,
          ),
          r"`allowed_batch_sizes` must be positive",
      ),
      (
          "main_invalid_allowed_batch_sizes_not_sorted",
          False,
          orchestration_pb2.BatchOptions(
              max_batch_size=8,
              allowed_batch_sizes=[8, 4, 2],
              max_enqueued_batches=1,
          ),
          "allowed_batch_sizes` must be sorted",
      ),
      (
          "main_invalid_max_batch_size_smaller_than_largest_allowed",
          False,
          orchestration_pb2.BatchOptions(
              max_batch_size=8,
              allowed_batch_sizes=[2, 4, 16],
              max_enqueued_batches=1,
          ),
          r"`max_batch_size` must be larger than or equal to the largest one",
      ),
      (
          "main_invalid_max_batch_size_not_equal_to_largest_allowed_splitting_disabled",
          False,
          orchestration_pb2.BatchOptions(
              max_batch_size=4,
              allowed_batch_sizes=[2, 8],
              disable_large_batch_splitting=True,
              max_enqueued_batches=1,
          ),
          "max_batch_size` must be equal to the largest one",
      ),
      (
          "main_invalid_batch_timeout_micros",
          False,
          orchestration_pb2.BatchOptions(
              max_batch_size=8, batch_timeout_micros=-1, max_enqueued_batches=1
          ),
          r"`batch_timeout_micros` must be non-negative",
      ),
      (
          "main_invalid_max_enqueued_batches",
          False,
          orchestration_pb2.BatchOptions(
              max_batch_size=8, max_enqueued_batches=0
          ),
          r"`max_enqueued_batches` must be at least 1",
      ),
      (
          "low_priority_missing_max_batch_size",
          True,
          orchestration_pb2.BatchOptions.LowPriorityBatchOptions(
              max_enqueued_batches=1
          ),
          "Low priority max_batch_size must be provided",
      ),
      (
          "low_priority_invalid_max_batch_size_negative",
          True,
          orchestration_pb2.BatchOptions.LowPriorityBatchOptions(
              max_batch_size=-1, max_enqueued_batches=1
          ),
          "low_priority_batch_options.max_batch_size` must be positive",
      ),
      (
          "low_priority_invalid_allowed_batch_sizes_not_positive",
          True,
          orchestration_pb2.BatchOptions.LowPriorityBatchOptions(
              max_batch_size=16,
              allowed_batch_sizes=[0, 4, 16],
              max_enqueued_batches=1,
          ),
          r"`low_priority_batch_options.allowed_batch_sizes` must be positive",
      ),
      (
          "low_priority_invalid_allowed_batch_sizes_not_sorted",
          True,
          orchestration_pb2.BatchOptions.LowPriorityBatchOptions(
              max_batch_size=16,
              allowed_batch_sizes=[4, 2, 16],
              max_enqueued_batches=1,
          ),
          r"`low_priority_batch_options.allowed_batch_sizes` must be sorted",
      ),
      (
          "low_priority_invalid_max_batch_size_not_equal_to_largest_allowed_splitting_disabled",
          True,
          orchestration_pb2.BatchOptions.LowPriorityBatchOptions(
              max_batch_size=8,
              allowed_batch_sizes=[2, 4, 16],
              max_enqueued_batches=1,
          ),
          (
              r"`low_priority_batch_options.max_batch_size` must be equal to"
              r" the largest one"
          ),
          # This one needs disable_large_batch_splitting=True on the main options
      ),
      (
          "low_priority_invalid_max_batch_size_smaller_than_largest_allowed",
          True,
          orchestration_pb2.BatchOptions.LowPriorityBatchOptions(
              max_batch_size=8,
              allowed_batch_sizes=[2, 4, 16],
              max_enqueued_batches=1,
          ),
          (
              r"`low_priority_batch_options.max_batch_size` must be larger than"
              r" or equal to the largest one"
          ),
      ),
      (
          "low_priority_invalid_batch_timeout_micros",
          True,
          orchestration_pb2.BatchOptions.LowPriorityBatchOptions(
              max_batch_size=16, batch_timeout_micros=-1, max_enqueued_batches=1
          ),
          (
              r"`low_priority_batch_options.batch_timeout_micros` must be"
              r" non-negative"
          ),
      ),
      (
          "low_priority_invalid_max_enqueued_batches",
          True,
          orchestration_pb2.BatchOptions.LowPriorityBatchOptions(
              max_batch_size=16, max_enqueued_batches=0
          ),
          (
              r"`low_priority_batch_options.max_enqueued_batches` must be at"
              r" least 1"
          ),
      ),
      (
          "main_invalid_num_batch_threads_negative",
          False,
          orchestration_pb2.BatchOptions(
              max_batch_size=8, num_batch_threads=-1, max_enqueued_batches=1
          ),
          r"num_batch_threads` must be at least 1",
      ),
  )
  def test_batch_options_invalid(
      self, is_low_priority, options, error_regex, disable_splitting=False
  ):
    if is_low_priority:
      # Some cases need splitting disabled to trigger specific error
      if (
          "splitting_disabled" in self._testMethodName
          or "max_batch_size_not_equal_to_largest_allowed"
          in self._testMethodName
      ):
        disable_splitting = True

      batch_options = orchestration_pb2.BatchOptions(
          batch_component=orchestration_pb2.BatchOptions.MODEL_FUNCTION,
          max_batch_size=16,
          low_priority_batch_options=options,
          max_enqueued_batches=1,
          disable_large_batch_splitting=disable_splitting,
      )
    else:
      batch_options = options
      if isinstance(batch_options, orchestration_pb2.BatchOptions):
        batch_options.batch_component = (
            orchestration_pb2.BatchOptions.MODEL_FUNCTION
        )

    with self.assertRaisesRegex(ValueError, error_regex):
      serving_config.ServingConfig(batch_options=batch_options)


if __name__ == "__main__":
  absltest.main()
