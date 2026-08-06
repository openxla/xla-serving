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

"""Serving configuration for JAX models."""

from collections.abc import Sequence
import dataclasses
import itertools

from absl import logging
import jaxtyping

from neptune_model._src.core import shlo_type
from neptune_model._src.export.constants import DEFAULT_METHOD_KEY
from neptune_model.protos import orchestration_pb2


@dataclasses.dataclass(frozen=True)
class ModelCall:
  """Configuration for invoking a specific model method within an orchestration.

  Attributes:
    model_name: The name of the model to call.
    method_key: The identifier of the method to invoke on the model.
    input_keys: The keys of the input dict that the function expects. Only used
      when the `graph_based` flag is True.
    output_keys: The keys of the output dict that the function produces. Only
      used when the `graph_based` flag is True.
    use_batching: Whether to enable batching for this call.
    role: Optional component role in the pipeline (e.g. ROLE_PREPROCESSOR,
      ROLE_MODEL, ROLE_POSTPROCESSOR). If None, defaults to ROLE_MODEL.
  """

  model_name: str
  method_key: str = DEFAULT_METHOD_KEY
  input_keys: Sequence[str] = dataclasses.field(default_factory=list)
  output_keys: Sequence[str] = dataclasses.field(default_factory=list)
  use_batching: bool = False
  role: orchestration_pb2.Role | None = orchestration_pb2.ROLE_MODEL


@dataclasses.dataclass(frozen=True)
class Orchestration:
  """Defines an ordered or dependency-driven workflow of model invocations.

  Attributes:
    model_calls: The sequence of model calls to execute.
    graph_based: If False, calls are executed sequentially in order. If True,
      execution order is determined by data dependencies.
  """

  model_calls: Sequence[ModelCall]
  graph_based: bool = False


@dataclasses.dataclass(kw_only=True)
class ServingConfig:
  """Serving config for a Jax model.

  Attributes:
    serving_signature: The name that uniquely identifies this serving config in
      the serving pipeline. Defaults to "serving_default".
    input_signature: The expected structure and shape of inputs for the model,
      represented as a JAX PyTree.
    batch_options: Optional batching configuration to use when serving the
      model.
    orchestration: Optional workflow defining a sequence or dependency graph of
      model invocations.
    output_placements: Optional placement configuration for output tensors.
  """

  serving_signature: str = "serving_default"
  # TODO: Evaluate if we need to make this a sequence of PyTrees.
  input_signature: jaxtyping.PyTree | None = None
  batch_options: orchestration_pb2.BatchOptions | None = None
  orchestration: Orchestration | None = None
  output_placements: (
      jaxtyping.PyTree[shlo_type.OutputPlacement]
      | Sequence[shlo_type.OutputPlacement]
      | None
  ) = None

  def __post_init__(self):
    # TODO (b/534785167): Update batch config validation logic.
    """Validates the batch options."""

    if self.batch_options is None or self.batch_options.batch_component in (
        orchestration_pb2.BatchOptions.BATCH_COMPONENT_UNSPECIFIED,
        orchestration_pb2.BatchOptions.NO_BATCHING,
    ):
      return

    if self.batch_options.max_batch_size == 0:
      if self.batch_options.allowed_batch_sizes:
        self.batch_options.max_batch_size = (
            self.batch_options.allowed_batch_sizes[-1]
        )
      else:
        raise ValueError(
            "max_batch_size must be provided when allowed_batch_sizes is"
            f" empty. Got: max_batch_size: {self.batch_options.max_batch_size};"
            f" allowed_batch_sizes: {self.batch_options.allowed_batch_sizes}."
        )

    _validate_batch_options(
        self.batch_options.max_batch_size,
        self.batch_options.allowed_batch_sizes,
        self.batch_options.batch_timeout_micros,
        self.batch_options.max_enqueued_batches,
        self.batch_options.disable_large_batch_splitting,
    )

    if self.batch_options.num_batch_threads < 0:
      raise ValueError(
          "`num_batch_threads` must be at least 1. Got:"
          f" {self.batch_options.num_batch_threads}"
      )
    elif self.batch_options.num_batch_threads == 0:
      # Set num_batch_threads to 1 if it is 0.
      logging.warning(
          "num_batch_threads is %d. Setting it to 1.",
          self.batch_options.num_batch_threads,
      )
      self.batch_options.num_batch_threads = 1

    if self.batch_options.HasField("low_priority_batch_options"):
      low_priority_batch_options = self.batch_options.low_priority_batch_options
      # Proto fields use 0 as default, not None.
      if low_priority_batch_options.max_batch_size == 0:
        if low_priority_batch_options.allowed_batch_sizes:
          low_priority_batch_options.max_batch_size = (
              low_priority_batch_options.allowed_batch_sizes[-1]
          )
        else:
          raise ValueError(
              "Low priority max_batch_size must be provided when"
              " allowed_batch_sizes is empty. Got:"
              " low_priority_batch_options.max_batch_size:"
              f" {low_priority_batch_options.max_batch_size};"
              " low_priority_batch_options.allowed_batch_sizes:"
              f" {low_priority_batch_options.allowed_batch_sizes}."
          )
      _validate_batch_options(
          low_priority_batch_options.max_batch_size,
          low_priority_batch_options.allowed_batch_sizes,
          low_priority_batch_options.batch_timeout_micros,
          low_priority_batch_options.max_enqueued_batches,
          self.batch_options.disable_large_batch_splitting,
          is_low_priority_batch_options=True,
      )


def _validate_batch_options(
    max_batch_size: int,
    allowed_batch_sizes: Sequence[int] | None,
    batch_timeout_micros: int,
    max_enqueued_batches: int,
    disable_large_batch_splitting: bool,
    is_low_priority_batch_options: bool = False,
) -> None:
  """Validates the batch options.

  Args:
    max_batch_size: The max batch size.
    allowed_batch_sizes: The allowed batch sizes.
    batch_timeout_micros: The batch timeout in microseconds.
    max_enqueued_batches: The max number of enqueued batches.
    disable_large_batch_splitting: Whether to disable large batch splitting.
    is_low_priority_batch_options: Whether the batch options are for low
      priority batches.

  Raises:
    ValueError: If the batch options are invalid.
  """
  low_priority_prefix = (
      "low_priority_batch_options." if is_low_priority_batch_options else ""
  )
  if max_batch_size < 1:
    raise ValueError(
        f"`{low_priority_prefix}max_batch_size` must be positive. Got:"
        f" {max_batch_size}"
    )
  if allowed_batch_sizes:
    # The allowed batch sizes must be positive and sorted in ascending order.
    for a, b in itertools.pairwise(allowed_batch_sizes):
      if a >= b:
        raise ValueError(
            f"`{low_priority_prefix}allowed_batch_sizes` must be sorted in"
            f" ascending order. Got: {allowed_batch_sizes}"
        )
    if allowed_batch_sizes[0] < 1:
      raise ValueError(
          f"`{low_priority_prefix}allowed_batch_sizes` must be positive. Got:"
          f" {allowed_batch_sizes}"
      )

    # When `allowed_batch_sizes` and `max_batch_size` are provided,
    # `max_batch_size` must be larger than or equal to the largest allowed
    # batch size.
    if (
        disable_large_batch_splitting
        and max_batch_size != allowed_batch_sizes[-1]
    ):
      raise ValueError(
          f"`{low_priority_prefix}max_batch_size` must be equal to the"
          f" largest one in `{low_priority_prefix}allowed_batch_sizes` when"
          f" large batch splitting is disabled. Got: {max_batch_size} vs"
          f" {allowed_batch_sizes[-1]}. Set"
          f" `{low_priority_prefix}max_batch_size` to None to automatically"
          f" infer it from `{low_priority_prefix}allowed_batch_sizes`."
      )
    if (
        not disable_large_batch_splitting
        and max_batch_size < allowed_batch_sizes[-1]
    ):
      raise ValueError(
          f"`{low_priority_prefix}max_batch_size` must be larger than or"
          " equal to the largest one in"
          f" `{low_priority_prefix}allowed_batch_sizes`. Got:"
          f" {max_batch_size} vs {allowed_batch_sizes[-1]}."
      )

  if batch_timeout_micros < 0:
    raise ValueError(
        f"`{low_priority_prefix}batch_timeout_micros` must be non-negative."
        f" Got: {batch_timeout_micros}"
    )

  if max_enqueued_batches <= 0:
    raise ValueError(
        f"`{low_priority_prefix}max_enqueued_batches` must be at least 1."
        f" Got: {max_enqueued_batches}"
    )
