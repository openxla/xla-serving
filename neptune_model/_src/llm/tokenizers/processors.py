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

"""Pre-processing and Post-processing tokenizers for LLM serving."""

import json
from typing import Any
from absl import logging
import keras_hub
from keras_hub.src.tokenizers import byte_pair_tokenizer
import tensorflow as tf
import transformers

AutoTokenizer = transformers.AutoTokenizer


def truncate_at_eos(token_ids: tf.Tensor, eos_token_id: int) -> tf.Tensor:
  """Truncates token IDs at the first EOS token index.

  Args:
    token_ids: Input tensor of token IDs.
    eos_token_id: Integer ID representing end of sequence.

  Returns:
    RaggedTensor of truncated token IDs.
  """
  is_eos = tf.equal(token_ids, eos_token_id)
  eos_cum = tf.cumsum(tf.cast(is_eos, tf.int32), axis=-1)
  keep_mask = tf.equal(eos_cum, 0)
  return tf.ragged.boolean_mask(token_ids, keep_mask)


def _resolve_tokenizer_class(hf_class_name: str, model_name_or_path: str):
  """Resolves the Keras Hub tokenizer class from the HF class or model path.

  Args:
    hf_class_name: Name of the Hugging Face tokenizer class.
    model_name_or_path: Hugging Face model identifier or file path.

  Returns:
    The corresponding Keras Hub tokenizer class.
  """
  # Normalize inputs
  norm_class_name = hf_class_name.replace("Fast", "").lower()
  model_name_lower = model_name_or_path.lower()

  # Map normalized Hugging Face tokenizer classes to Keras Hub implementations
  tokenizer_mapping = {
      "qwen3tokenizer": keras_hub.tokenizers.Qwen3Tokenizer,
      "qwen2tokenizer": getattr(
          keras_hub.tokenizers,
          "Qwen2Tokenizer",
          keras_hub.tokenizers.Qwen3Tokenizer,
      ),
      "llama3tokenizer": keras_hub.tokenizers.Llama3Tokenizer,
      "llamatokenizer": keras_hub.tokenizers.LlamaTokenizer,
      "gpt2tokenizer": keras_hub.tokenizers.GPT2Tokenizer,
  }

  if norm_class_name in tokenizer_mapping:
    return tokenizer_mapping[norm_class_name]

  if "qwen3" in norm_class_name or "qwen3" in model_name_lower:
    return keras_hub.tokenizers.Qwen3Tokenizer
  elif (
      "qwen2" in norm_class_name
      or "qwen" in norm_class_name
      or "qwen" in model_name_lower
  ):
    return getattr(
        keras_hub.tokenizers,
        "Qwen2Tokenizer",
        keras_hub.tokenizers.Qwen3Tokenizer,
    )
  elif (
      "llama3" in norm_class_name
      or "llama-3" in model_name_lower
      or "llama3" in model_name_lower
  ):
    return keras_hub.tokenizers.Llama3Tokenizer
  elif "llama" in norm_class_name or "llama" in model_name_lower:
    return keras_hub.tokenizers.LlamaTokenizer
  elif "gpt2" in norm_class_name or "gpt2" in model_name_lower:
    return keras_hub.tokenizers.GPT2Tokenizer

  print(
      f"Fallback to generic BytePairTokenizer for class {hf_class_name} / model"
      f" {model_name_or_path}"
  )
  return keras_hub.tokenizers.BytePairTokenizer


class KerasHubTokenizerTF:
  """A graph-compatible pre-processor and post-processor wrapping Keras Hub."""

  def __init__(
      self,
      model_name_or_path: str,
      eos_token_id: int | None = None,
      pad_token_id: int | None = None,
      max_seq_len: int = 64,
      max_candidate_tokens: int | None = None,
  ):
    """Initializes the Keras Hub TF tokenizer.

    Args:
      model_name_or_path: Hugging Face model identifier or directory path.
      eos_token_id: End of sequence token ID. If None, pulled from tokenizer.
      pad_token_id: Padding token ID. If None, pulled from tokenizer.
      max_seq_len: Maximum sequence length to pad or slice to.
      max_candidate_tokens: Maximum token length for candidate completions.
    """
    self.model_name_or_path = model_name_or_path
    self.max_seq_len = max_seq_len
    self.max_candidate_tokens = max_candidate_tokens

    # Load Hugging Face tokenizer to extract vocab and merges
    print(f"Loading Hugging Face tokenizer for {model_name_or_path}...")
    real_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    self.pad_token_id = (
        pad_token_id
        if pad_token_id is not None
        else real_tokenizer.pad_token_id  # pyrefly: ignore[missing-attribute]
    )
    if self.pad_token_id is None:
      raise ValueError("pad_token_id could not be resolved from tokenizer")

    self.eos_token_id = (
        eos_token_id
        if eos_token_id is not None
        else real_tokenizer.eos_token_id  # pyrefly: ignore[missing-attribute]
    )
    if self.eos_token_id is None:
      raise ValueError("eos_token_id could not be resolved from tokenizer")

    vocab = real_tokenizer.get_vocab()  # pyrefly: ignore[missing-attribute]
    vocab = {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1])}

    tokenizer_data = json.loads(real_tokenizer._tokenizer.to_str())  # pyrefly: ignore[missing-attribute]
    merges = tokenizer_data.get("model", {}).get("merges", [])
    formatted_merges = [" ".join(pair) for pair in merges]

    hf_class_name = real_tokenizer.__class__.__name__
    logging.info("Detected Hugging Face tokenizer class: %s", hf_class_name)
    tokenizer_cls = _resolve_tokenizer_class(hf_class_name, model_name_or_path)
    logging.info(
        "Selected Keras Hub tokenizer class: %s", tokenizer_cls.__name__
    )

    self.tokenizer = tokenizer_cls(
        vocabulary=vocab,
        merges=formatted_merges,
    )
    # Warmup / build all tokenizer tables so StaticHashTables are constructed
    _ = self.tokenizer.detokenize(tf.constant([[1, 2]], dtype=tf.int32))

  def _tokenize(self, text_input: tf.Tensor) -> tf.RaggedTensor:
    """Tokenizes text input statelessly without mutable cache resources.

    Adapted from KerasHub's `BytePairTokenizer._tokenize_tf` (see
    `third_party/py/keras_hub/src/tokenizers/byte_pair_tokenizer.py:712-770`).
    Upstream `BytePairTokenizer` uses an internal `BytePairTokenizerCache`
    backed by `tf.lookup.experimental.MutableHashTable`. In SavedModel export
    and serving, this mutable cache creates untracked resource capture errors
    at runtime.

    We bypass the cache and run the BPE merge pipeline directly using
    KerasHub's static lookup tables (`byte2unicode`, `token_to_id_map`,
    `merge_ranks`) via `_bpe_merge_tf` and `split_strings_for_bpe`. For
    non-BPE tokenizers, this delegates to `self.tokenizer(text_input)`.

    Args:
      text_input: Input string tensor of rank 0 or 1.

    Returns:
      RaggedTensor containing token IDs or string tokens.
    """
    tokenizer = self.tokenizer
    # Fallback to generic tokenizer if BPE-specific attributes are not found.
    if not hasattr(tokenizer, "_bpe_merge_tf") or not hasattr(
        tokenizer, "token_to_id_map"
    ):
      return self.tokenizer(text_input)

    if getattr(tokenizer, "add_prefix_space", False):
      text_input = tf.strings.join([" ", text_input])

    text_input = tf.convert_to_tensor(text_input)
    unbatched = text_input.shape.rank == 0
    if unbatched:
      text_input = tf.expand_dims(text_input, 0)
    if text_input.shape.rank is not None and text_input.shape.rank > 1:
      raise ValueError(
          "`tokenize()` inputs should be a string, list of strings, or "
          f"string tensor with rank < 2. Received: {text_input}"
      )

    unsplittable_tokens = getattr(tokenizer, "unsplittable_tokens", None)
    raw_tokens = byte_pair_tokenizer.split_strings_for_bpe(
        text_input, unsplittable_tokens
    )
    token_row_splits = raw_tokens.row_splits
    flat_tokens = raw_tokens.flat_values

    split_bytes = tf.strings.bytes_split(flat_tokens)
    split_unicode = tokenizer.byte2unicode.lookup(split_bytes)
    # Access KerasHub's stateless BPE merge implementation.
    merged_words = tokenizer._bpe_merge_tf(split_unicode)  # pylint: disable=protected-access
    tokenized_words = tf.strings.reduce_join(
        merged_words, axis=1, separator=" "
    )

    tokens = tf.strings.split(tokenized_words, sep=" ")
    if getattr(tokenizer, "compute_dtype", None) != tf.string:
      tokens = tokenizer.token_to_id_map.lookup(tokens)

    tokens = tf.RaggedTensor.from_row_splits(
        tokens.flat_values,
        tf.gather(tokens.row_splits, token_row_splits),
    )
    if getattr(tokenizer, "sequence_length", None):
      output_shape = tokens.shape.as_list()
      output_shape[-1] = tokenizer.sequence_length
      tokens = tokens.to_tensor(shape=output_shape)
    if unbatched:
      tokens = tf.squeeze(tokens, 0)
    return tokens

  def pre_process_fn(
      self, inputs: dict[str, tf.Tensor]
  ) -> dict[str, tf.Tensor]:
    """Tokenizes text input inside TF graph.

    Args:
      inputs: Dictionary containing "text_input" string tensor and optional
        "candidate_texts" string tensor.

    Returns:
      Dictionary containing "input_ids" and optional "candidate_ids" integer
      tensors.
    """
    result = {}
    if "text_input" in inputs:
      text_input = inputs["text_input"]
      ragged_tokens = self._tokenize(text_input)
      if isinstance(ragged_tokens, tf.RaggedTensor):
        input_ids = ragged_tokens.to_tensor(default_value=self.pad_token_id)
      else:
        input_ids = tf.convert_to_tensor(ragged_tokens)

      unbatched = text_input.shape.rank == 0
      if unbatched and input_ids.shape.rank == 1:
        input_ids = tf.expand_dims(input_ids, 0)

      # Pad/slice input_ids to exactly max_seq_len
      paddings = [[0, 0], [0, self.max_seq_len]]
      padded = tf.pad(input_ids, paddings, constant_values=self.pad_token_id)
      input_ids_fixed = tf.slice(padded, [0, 0], [-1, self.max_seq_len])

      if unbatched:
        input_ids_fixed = tf.squeeze(input_ids_fixed, 0)
      elif text_input.shape[0] is not None:
        input_ids_fixed.set_shape([text_input.shape[0], self.max_seq_len])

      result["input_ids"] = tf.cast(input_ids_fixed, tf.int32)

    if "candidate_texts" in inputs:
      if self.max_candidate_tokens is None:
        raise ValueError(
            "max_candidate_tokens must be specified on KerasHubTokenizerTF"
            " when candidate_texts are provided."
        )
      cand_text = inputs["candidate_texts"]
      unbatched_cand = cand_text.shape.rank == 1
      if unbatched_cand:
        cand_text = tf.expand_dims(cand_text, 0)
      batch_dim = tf.shape(cand_text)[0]
      num_cands = tf.shape(cand_text)[1]
      flat_cands = tf.reshape(cand_text, [-1])
      ragged_cand = self._tokenize(flat_cands)
      if isinstance(ragged_cand, tf.RaggedTensor):
        cand_ids = ragged_cand.to_tensor(default_value=self.pad_token_id)
      else:
        cand_ids = tf.convert_to_tensor(ragged_cand)

      max_cand_len = self.max_candidate_tokens
      paddings_cand = [[0, 0], [0, max_cand_len]]
      padded_cand = tf.pad(
          cand_ids, paddings_cand, constant_values=self.pad_token_id
      )
      cand_ids_fixed = tf.slice(padded_cand, [0, 0], [-1, max_cand_len])
      cand_ids_3d = tf.reshape(
          cand_ids_fixed, [batch_dim, num_cands, max_cand_len]
      )
      if unbatched_cand:
        cand_ids_3d = tf.squeeze(cand_ids_3d, 0)
      elif cand_text.shape[0] is not None and cand_text.shape[1] is not None:
        cand_ids_3d.set_shape(
            [cand_text.shape[0], cand_text.shape[1], max_cand_len]
        )
      result["candidate_ids"] = tf.cast(cand_ids_3d, tf.int32)

    return result

  def post_process_fn(
      self, inputs: dict[str, tf.Tensor]
  ) -> dict[str, tf.Tensor]:
    """Detokenizes token ids or passes through scores inside TF graph.

    Args:
      inputs: Dictionary containing "token_ids" or "scores" tensors.

    Returns:
      Dictionary containing "text_output" string tensor or "scores" tensor.
    """
    if "token_ids" not in inputs:
      return inputs

    token_ids = inputs["token_ids"]

    ndims = token_ids.shape.ndims
    if ndims == 3:
      # Beam search output shape: (batch_size, beam_width, max_new_tokens)
      batch_size = tf.shape(token_ids)[0]
      beam_width = tf.shape(token_ids)[1]
      max_new_tokens = tf.shape(token_ids)[2]

      flat_ids = tf.reshape(token_ids, [-1, max_new_tokens])
      truncated_ids = truncate_at_eos(flat_ids, self.eos_token_id)
      flat_text = self.tokenizer.detokenize(truncated_ids)
      text_output = tf.reshape(flat_text, [batch_size, beam_width])
    else:
      truncated_ids = truncate_at_eos(token_ids, self.eos_token_id)
      text_output = self.tokenizer.detokenize(truncated_ids)

    if not isinstance(text_output, tf.Tensor):
      text_output = tf.constant(text_output)
    return {"text_output": text_output}

  def get_trackable_resources(self) -> list[Any]:
    """Returns static lookup tables for SavedModel export, omitting self.cache."""
    resources = []
    t = self.tokenizer
    if hasattr(t, "token_to_id_map") and t.token_to_id_map is not None:
      resources.append(t.token_to_id_map)
    if hasattr(t, "id_to_token_map") and t.id_to_token_map is not None:
      resources.append(t.id_to_token_map)
    if hasattr(t, "unicode2byte") and t.unicode2byte is not None:
      resources.append(t.unicode2byte)
    if hasattr(t, "byte2unicode") and t.byte2unicode is not None:
      resources.append(t.byte2unicode)
    if hasattr(t, "merge_ranks") and t.merge_ranks is not None:
      resources.append(t.merge_ranks)
    return resources
