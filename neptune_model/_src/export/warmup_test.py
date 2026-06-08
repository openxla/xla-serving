# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under/portions of the License.

import os
import struct

from absl.testing import absltest
import numpy as np

from neptune_model._src.export import warmup
from neptune_model.protos.grpc import prediction_log_pb2 as neptune_prediction_log_pb2


def _read_size_prefixed_records(file_path: str) -> list[bytes]:
  records = []
  with open(file_path, "rb") as f:
    while True:
      size_bytes = f.read(4)
      if not size_bytes:
        break
      if len(size_bytes) < 4:
        raise ValueError("Truncated size prefix")
      size = struct.unpack("<I", size_bytes)[0]
      record = f.read(size)
      if len(record) < size:
        raise ValueError("Truncated record payload")
      records.append(record)
  return records


class WarmupTest(absltest.TestCase):

  def test_write_warmup_file(self):
    model_dir = self.create_tempdir().full_path
    warmup_examples = {
        "serving_default": [
            {"x": np.ones((2, 3), dtype=np.float32), "s": np.array(["hello"])},
            {"x": np.zeros((2, 3), dtype=np.float32), "s": np.array(["world"])},
        ],
        "other_signature": [
            {"y": np.zeros((1, 2), dtype=np.int32)},
            {"y": np.ones((1, 2), dtype=np.int32)},
        ],
    }
    warmup.write_warmup_file(warmup_examples, model_dir)

    warmup_path = os.path.join(
        model_dir, warmup.ASSETS_EXTRA_DIR, warmup.WARMUP_REQUESTS_FILENAME
    )
    self.assertTrue(os.path.exists(warmup_path))

    # Verify content using direct size-prefixed binary reader
    records = _read_size_prefixed_records(warmup_path)
    self.assertLen(records, 4)

    log_serving_default_1 = neptune_prediction_log_pb2.PredictionLog()
    log_serving_default_1.ParseFromString(records[0])
    self.assertEqual(
        log_serving_default_1.predict_log.request.model_spec.signature_name,
        "serving_default",
    )
    self.assertIn("x", log_serving_default_1.predict_log.request.inputs)
    self.assertIn("s", log_serving_default_1.predict_log.request.inputs)

    log_serving_default_2 = neptune_prediction_log_pb2.PredictionLog()
    log_serving_default_2.ParseFromString(records[1])
    self.assertEqual(
        log_serving_default_2.predict_log.request.model_spec.signature_name,
        "serving_default",
    )
    self.assertIn("x", log_serving_default_2.predict_log.request.inputs)
    self.assertIn("s", log_serving_default_2.predict_log.request.inputs)

    log_other_signature_1 = neptune_prediction_log_pb2.PredictionLog()
    log_other_signature_1.ParseFromString(records[2])
    self.assertEqual(
        log_other_signature_1.predict_log.request.model_spec.signature_name,
        "other_signature",
    )
    self.assertIn("y", log_other_signature_1.predict_log.request.inputs)

    log_other_signature_2 = neptune_prediction_log_pb2.PredictionLog()
    log_other_signature_2.ParseFromString(records[3])
    self.assertEqual(
        log_other_signature_2.predict_log.request.model_spec.signature_name,
        "other_signature",
    )
    self.assertIn("y", log_other_signature_2.predict_log.request.inputs)


if __name__ == "__main__":
  absltest.main()
