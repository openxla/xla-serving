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

import collections
from collections.abc import Mapping, Sequence
import dataclasses
import itertools

from absl import logging
import jaxtyping

from xla_saved_model.protos import orchestration_pb2
from xla_saved_model._src.export.constants import DEFAULT_METHOD_KEY


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


def topological_sort(
    model_calls: Sequence[ModelCall],
) -> Sequence[ModelCall]:
  """Topologically sorts model calls based on input and output key dependencies.

  Args:
    model_calls: The sequence of ModelCall objects to sort.

  Returns:
    A sequence of ModelCall objects in topological execution order.

  Raises:
    ValueError: If any ModelCall is missing input_keys or output_keys.
    ValueError: If an output key is produced by multiple ModelCalls.
    ValueError: If there is a cycle in the dependency graph.
  """
  for call in model_calls:
    if not call.input_keys:
      raise ValueError(
          f"ModelCall for model {call.model_name!r} and method"
          f" {call.method_key!r} must have `input_keys` set when"
          " `graph_based` is True."
      )
    if not call.output_keys:
      raise ValueError(
          f"ModelCall for model {call.model_name!r} and method"
          f" {call.method_key!r} must have `output_keys` set when"
          " `graph_based` is True."
      )

  output_name_to_producer_indices = collections.defaultdict(list)
  for i, call in enumerate(model_calls):
    for name in call.output_keys:
      output_name_to_producer_indices[name].append(i)

  for name, indices in output_name_to_producer_indices.items():
    if len(indices) > 1:
      producers = [
          f"ModelCall(model_name={model_calls[i].model_name!r},"
          f" method_key={model_calls[i].method_key!r})"
          for i in indices
      ]
      raise ValueError(
          f"Output key '{name}' is produced by multiple model calls:"
          f" {producers}"
      )

  adj = collections.defaultdict(list)
  in_degree = collections.defaultdict(int)
  for i, call in enumerate(model_calls):
    for name in call.input_keys:
      if name in output_name_to_producer_indices:
        for producer_idx in output_name_to_producer_indices[name]:
          adj[producer_idx].append(i)
          in_degree[i] += 1

  queue = collections.deque(
      [i for i in range(len(model_calls)) if in_degree[i] == 0]
  )
  sorted_calls: list[ModelCall] = []

  while queue:
    node_idx = queue.popleft()
    sorted_calls.append(model_calls[node_idx])
    for neighbor_idx in adj[node_idx]:
      in_degree[neighbor_idx] -= 1
      if in_degree[neighbor_idx] == 0:
        queue.append(neighbor_idx)

  if len(sorted_calls) != len(model_calls):
    raise ValueError("Dependency graph has a cycle.")

  return sorted_calls


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
  _ordered_model_calls: Sequence[ModelCall] = dataclasses.field(
      init=False, repr=False, compare=False
  )

  def __post_init__(self):
    """Validates the orchestration configuration and caches execution order."""
    if self.graph_based:
      object.__setattr__(
          self, "_ordered_model_calls", topological_sort(self.model_calls)
      )
    else:
      object.__setattr__(self, "_ordered_model_calls", self.model_calls)

    # Check that batching calls are consecutive.
    batching_groups = [
        k
        for k, _ in itertools.groupby(
            self._ordered_model_calls, key=lambda c: c.use_batching
        )
    ]
    if batching_groups.count(True) > 1:
      raise ValueError(
          "Model calls with `use_batching=True` must be consecutive in the"
          " pipeline."
      )

  @property
  def ordered_model_calls(self) -> Sequence[ModelCall]:
    """Returns model calls in execution order.

    If `graph_based` is True, returns model calls sorted topologically
    based on data dependencies (input_keys and output_keys). Otherwise,
    returns `model_calls` in sequential order.
    """
    return self._ordered_model_calls


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
  """

  serving_signature: str = "serving_default"
  # TODO: Evaluate if we need to make this a sequence of PyTrees.
  input_signature: jaxtyping.PyTree = None
  batch_options: orchestration_pb2.BatchOptions | None = None
  orchestration: Orchestration | None = None

  def __post_init__(self):
    # TODO: Update batch config validation logic.
    """Validates the batch options and orchestration."""
    if self.orchestration is not None and self.orchestration.graph_based:
      if (
          not isinstance(self.input_signature, Mapping)
          or not self.input_signature
      ):
        raise ValueError(
            "When `graph_based` is True, `input_signature` must be a Mapping"
            " of input names to tensor specs and cannot be empty. Got:"
            f" {self.input_signature}"
        )

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

    if (
        self.batch_options.priority_batching_policy
        == orchestration_pb2.BatchOptions.PRIORITY_AWARE
        and self.batch_options.HasField("priority_aware_batching_options")
    ):
      priority_aware_batching_options = (
          self.batch_options.priority_aware_batching_options
      )
      if priority_aware_batching_options.max_queue_depth < 0:
        raise ValueError(
            "`priority_aware_batching_options.max_queue_depth` must be"
            " non-negative. Got:"
            f" {priority_aware_batching_options.max_queue_depth}"
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
