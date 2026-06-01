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
    warmup_examples = [
        {"x": np.ones((2, 3), dtype=np.float32), "s": np.array(["hello"])},
    ]
    warmup.write_warmup_file({"serving_default": warmup_examples}, model_dir)

    warmup_path = os.path.join(
        model_dir, "assets.extra", "tf_serving_warmup_requests"
    )
    self.assertTrue(os.path.exists(warmup_path))

    # Verify content using direct size-prefixed binary reader
    records = _read_size_prefixed_records(warmup_path)
    self.assertLen(records, 1)
    log = neptune_prediction_log_pb2.PredictionLog()
    log.ParseFromString(records[0])
    self.assertEqual(
        log.predict_log.request.model_spec.signature_name, "serving_default"
    )
    self.assertIn("x", log.predict_log.request.inputs)
    self.assertIn("s", log.predict_log.request.inputs)


if __name__ == "__main__":
  absltest.main()
