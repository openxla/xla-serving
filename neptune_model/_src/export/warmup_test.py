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

import io
import os

from absl.testing import absltest
from google.protobuf import proto
import numpy as np

from neptune_model._src.export import warmup
from neptune_model.protos.grpc import prediction_log_pb2 as neptune_prediction_log_pb2


def _read_warmup_records(
    file_path: str,
) -> list[neptune_prediction_log_pb2.PredictionLog]:
  records = []
  with open(file_path, "rb") as f:
    bytes_io = io.BytesIO(f.read())
    while True:
      record = proto.parse_length_prefixed(
          neptune_prediction_log_pb2.PredictionLog, bytes_io
      )
      if record is None:
        break
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

    # Verify content using standard proto parser
    records = _read_warmup_records(warmup_path)
    self.assertLen(records, 4)

    log_serving_default_1 = records[0]
    self.assertEqual(
        log_serving_default_1.predict_log.request.model_spec.signature_name,
        "serving_default",
    )
    self.assertIn("x", log_serving_default_1.predict_log.request.inputs)
    self.assertIn("s", log_serving_default_1.predict_log.request.inputs)

    log_serving_default_2 = records[1]
    self.assertEqual(
        log_serving_default_2.predict_log.request.model_spec.signature_name,
        "serving_default",
    )
    self.assertIn("x", log_serving_default_2.predict_log.request.inputs)
    self.assertIn("s", log_serving_default_2.predict_log.request.inputs)

    log_other_signature_1 = records[2]
    self.assertEqual(
        log_other_signature_1.predict_log.request.model_spec.signature_name,
        "other_signature",
    )
    self.assertIn("y", log_other_signature_1.predict_log.request.inputs)

    log_other_signature_2 = records[3]
    self.assertEqual(
        log_other_signature_2.predict_log.request.model_spec.signature_name,
        "other_signature",
    )
    self.assertIn("y", log_other_signature_2.predict_log.request.inputs)


if __name__ == "__main__":
  absltest.main()
