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
"""Tests for Neptune export of PyTorch model."""

import os
from absl.testing import absltest
import numpy as np
import torch
from examples.torchax import export
from examples.torchax import model as torch_model
from xla_saved_model_runner import model_runner


class ExportTest(absltest.TestCase):
  """Tests for Neptune export of PyTorch model."""

  def test_export_and_compare(self):
    export_path = self.create_tempdir().full_path

    pt_model = torch_model.get_model()
    export.export_model(pt_model, export_path)

    self.assertTrue(os.path.exists(export_path))
    self.assertTrue(os.path.isdir(os.path.join(export_path, "checkpoint")))
    self.assertTrue(
        os.path.isfile(os.path.join(export_path, "orchestration.pb"))
    )

    input_np = np.random.randn(1, 10).astype(np.float32)
    input_tensor = torch.from_numpy(input_np)

    pt_output = pt_model(input_tensor).detach().numpy()

    runner = model_runner.ModelRunner(model_path=export_path)
    neptune_outputs = runner.signature("serving_default").run_with_dict_return(
        x=input_np
    )
    neptune_output = neptune_outputs["y"]

    np.testing.assert_allclose(neptune_output, pt_output, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
