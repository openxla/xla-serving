# Copyright 2026 Arm Ltd. and affiliates
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

"""Compare all E5 serving signatures with an independent PyTorch reference."""

import numpy as np
import torch
from absl.testing import absltest
from torch.nn import functional
from transformers import AutoModel, AutoTokenizer
from xla_saved_model_runner import model_runner

from examples.e5 import export


class ExportTest(absltest.TestCase):
    def test_export_and_compare(self):
        export_path = self.create_tempdir().full_path
        export.export_model(export_path)
        runner = model_runner.ModelRunner(model_path=export_path, platform="cpu")
        model = (
            AutoModel.from_pretrained(
                export.DEFAULT_MODEL, revision=export.DEFAULT_REVISION
            )
            .float()
            .eval()
        )
        tokenizer = AutoTokenizer.from_pretrained(
            export.DEFAULT_MODEL, revision=export.DEFAULT_REVISION
        )
        texts = (
            "query: what causes a rainbow?",
            "passage: " + "Sunlight is refracted and reflected by water drops. " * 128,
        )
        for sequence_length in (128, 256, 512):
            signature = runner.signature(f"embed_b1_s{sequence_length}")
            for text in texts:
                with self.subTest(sequence_length=sequence_length, prefix=text[:8]):
                    encoded = tokenizer(
                        text,
                        padding="max_length",
                        truncation=True,
                        max_length=sequence_length,
                        return_tensors="np",
                    )
                    inputs = {
                        name: np.asarray(encoded[name], dtype=np.int32)
                        for name in ("input_ids", "attention_mask", "token_type_ids")
                    }
                    # Exercise both padding and a full-length (truncated) input.
                    token_count = int(inputs["attention_mask"].sum())
                    if text.startswith("query:"):
                        self.assertGreater(token_count, 0)
                        self.assertLess(token_count, sequence_length)
                    else:
                        self.assertEqual(token_count, sequence_length)

                    with torch.inference_mode():
                        tensors = {k: torch.from_numpy(v) for k, v in inputs.items()}
                        hidden = model(**tensors).last_hidden_state
                        hidden = hidden.masked_fill(
                            ~tensors["attention_mask"][..., None].bool(), 0.0
                        )
                        pooled = hidden.sum(dim=1) / tensors["attention_mask"].sum(
                            dim=1, keepdim=True
                        )
                        reference = functional.normalize(pooled, p=2, dim=1).numpy()

                    outputs = signature.run_with_dict_return(**inputs)
                    self.assertEqual(set(outputs), {"embeddings"})
                    actual = outputs["embeddings"]
                    for embeddings in (actual, reference):
                        self.assertEqual(embeddings.shape, (1, 768))
                        self.assertEqual(embeddings.dtype, np.float32)
                        self.assertTrue(np.isfinite(embeddings).all())

                    actual = actual.astype(np.float64)
                    reference = reference.astype(np.float64)
                    difference = np.abs(actual - reference)
                    self.assertLessEqual(difference.max(), 2e-4)
                    self.assertLessEqual(difference.mean(), 2e-5)
                    actual_norm = np.linalg.norm(actual, axis=1)
                    reference_norm = np.linalg.norm(reference, axis=1)
                    for norm in (actual_norm, reference_norm):
                        np.testing.assert_allclose(norm, 1.0, rtol=0, atol=2e-5)


if __name__ == "__main__":
    absltest.main()
