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

"""Unit tests for LLM tokenization processors."""

# pylint: disable=g-importing-member

import os

from absl.testing import absltest
import tensorflow as tf

from xla_saved_model._src.llm.tokenizers import processors


class ProcessorsTest(absltest.TestCase):

  def test_truncate_at_eos_2d(self):
    token_ids = tf.constant(
        [
            [10, 11, 2, 13, 14],
            [20, 21, 22, 2, 24],
        ],
        dtype=tf.int32,
    )
    truncated = processors.truncate_at_eos(token_ids, eos_token_id=2)
    self.assertIsInstance(truncated, tf.RaggedTensor)
    self.assertEqual(truncated[0].numpy().tolist(), [10, 11])
    self.assertEqual(truncated[1].numpy().tolist(), [20, 21, 22])

  def test_resolve_tokenizer_class(self):
    cls_qwen = processors._resolve_tokenizer_class(
        "Qwen2TokenizerFast", "Qwen/Qwen2-0.5B"
    )
    self.assertIn(
        cls_qwen.__name__, ("QwenTokenizer", "Qwen2Tokenizer", "Qwen3Tokenizer")
    )

    cls_fallback = processors._resolve_tokenizer_class(
        "UnknownTokenizer", "unknown/model"
    )
    self.assertEqual(cls_fallback.__name__, "BytePairTokenizer")

  def test_keras_hub_tokenizer_tf_instantiation_and_processing(self):
    model_path = os.path.join(
        os.path.dirname(__file__),
        "../testdata/qwen_dummy",
    )

    tokenizer = processors.KerasHubTokenizerTF(
        model_name_or_path=model_path,
        max_seq_len=16,
    )
    self.assertEqual(tokenizer.eos_token_id, 0)
    self.assertEqual(tokenizer.pad_token_id, 0)
    resources_list = tokenizer.get_trackable_resources()
    self.assertIsInstance(resources_list, list)

    inputs = {"text_input": tf.constant(["ab", "a"])}
    preprocess_result = tokenizer.pre_process_fn(inputs)
    self.assertEqual(list(preprocess_result.keys()), ["input_ids"])
    self.assertEqual(preprocess_result["input_ids"].shape, (2, 16))

    # Test 2D post-processing (detokenizing input_ids recovers original text)
    post_inputs_2d = {"token_ids": preprocess_result["input_ids"]}
    postprocess_result_2d = tokenizer.post_process_fn(post_inputs_2d)
    self.assertEqual(postprocess_result_2d["text_output"].shape, (2,))
    self.assertEqual(
        postprocess_result_2d["text_output"].numpy().tolist(), [b"ab", b"a"]
    )

    # Test 3D (beam search) post-processing with actual token ID 4 ("ab")
    post_inputs_3d = {"token_ids": tf.fill((2, 4, 16), 4)}
    postprocess_result_3d = tokenizer.post_process_fn(post_inputs_3d)
    self.assertEqual(postprocess_result_3d["text_output"].shape, (2, 4))
    self.assertEqual(
        postprocess_result_3d["text_output"][0, 0].numpy(), b"ab" * 16
    )

  def test_tokenize_scalar_and_batched(self):
    model_path = os.path.join(
        os.path.dirname(__file__),
        "../testdata/qwen_dummy",
    )
    tokenizer = processors.KerasHubTokenizerTF(
        model_name_or_path=model_path,
        max_seq_len=16,
        max_candidate_tokens=32,
    )
    # Test scalar input
    scalar_res = tokenizer.pre_process_fn({"text_input": tf.constant("ab")})
    self.assertEqual(scalar_res["input_ids"].shape, (16,))
    self.assertEqual(scalar_res["input_ids"][0].numpy(), 4)

    # Test batched input
    batched_res = tokenizer.pre_process_fn(
        {"text_input": tf.constant(["ab", "a"])}
    )
    self.assertEqual(batched_res["input_ids"].shape, (2, 16))
    self.assertEqual(batched_res["input_ids"][0, 0].numpy(), 4)
    self.assertEqual(batched_res["input_ids"][1, 0].numpy(), 1)

    # Test candidate text pre-processing and post-processing
    inputs_cands = {
        "text_input": tf.constant(["ab"]),
        "candidate_texts": tf.constant([["ab", "a", "b"]]),
    }
    cand_preprocess = tokenizer.pre_process_fn(inputs_cands)
    self.assertIn("input_ids", cand_preprocess)
    self.assertIn("candidate_ids", cand_preprocess)
    self.assertEqual(cand_preprocess["input_ids"].shape, (1, 16))
    self.assertEqual(cand_preprocess["candidate_ids"].shape, (1, 3, 32))

    scores_post = tokenizer.post_process_fn({
        "scores": tf.constant([[-1.2, -0.8, -2.5]]),
        "step0_logprobs": tf.constant([[-0.5, -0.3, -1.0]]),
        "continuation_scores": tf.constant([[-0.7, -0.5, -1.5]]),
    })
    self.assertIn("scores", scores_post)
    self.assertIn("step0_logprobs", scores_post)
    self.assertIn("continuation_scores", scores_post)
    self.assertEqual(scores_post["scores"].shape, (1, 3))
    self.assertEqual(scores_post["step0_logprobs"].shape, (1, 3))
    self.assertEqual(scores_post["continuation_scores"].shape, (1, 3))


if __name__ == "__main__":
  absltest.main()
