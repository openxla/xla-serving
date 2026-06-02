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

"""Neptune model processing pipelines describe how to orchestrate processors around a model."""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any, Callable, Protocol

import jaxtyping

from neptune_model._src.core import persistence_lib
from neptune_model._src.core import saveable
from neptune_model._src.core import value


class PreparedProcessor(Protocol):
  """A processor that is prepared for export."""

  @property
  def name(self) -> str:
    ...

  @property
  def input_signature(self) -> jaxtyping.PyTree:
    ...

  @property
  def output_signature(self) -> jaxtyping.PyTree:
    ...

  @property
  def weights_name(self) -> str | None:
    ...

  def to_npt_saveable(self) -> saveable.Saveable:
    ...

  def on_persist(
      self,
  ) -> (
      Callable[
          [str],
          Mapping[
              str, value.ExternalValue | persistence_lib.GlobalSupplemental
          ],
      ]
      | None
  ):
    """A callback that runs prior to saving the processor during the export.

    Returns:
      The function that can be used to save extra objects
      for the processor. The function takes the export path as an argument
      and returns a mapping of object names to ExternalValues that should be
      saved for this processor. Returns None if no extra objects should be
      saved.
    """
    ...


class ProcessorDefaults:
  """Mixin providing defaults for PreparedProcessor methods."""

  @property
  def weights_name(self) -> str | None:
    return None

  def on_persist(
      self,
  ) -> (
      Callable[
          [str],
          Mapping[
              str, value.ExternalValue | persistence_lib.GlobalSupplemental
          ],
      ]
      | None
  ):
    return None


@dataclasses.dataclass(frozen=True, kw_only=True)
class ProcessingOptions:
  """Options for configuring the processing pipeline.

  Attributes:
    tf_trackable_resources: Optional TensorFlow trackable resources that should
      be included in the exported TF saved model. Relevant only if TF processors
      are used in the processing pipeline.
  """

  tf_trackable_resources: Any | None = None


class Processor(Protocol):
  """A processor as it's defined by the client."""

  def prepare(
      self,
      input_signature: jaxtyping.PyTree,
      **kwargs: Any,
  ) -> PreparedProcessor:
    """Prepares the processor for export."""
    ...


class Processing(Protocol):
  """A processing pipeline that runs processors in sequence."""

  @property
  def options(self) -> ProcessingOptions:
    """Returns the options for the processor."""
    ...


@dataclasses.dataclass(frozen=True)
class SequentialProcessing:
  """A processing pipeline that runs processors in sequence.

  This processing pipeline first runs the preprocessors in sequence, followed by
  the model, and then the postprocessors in sequence. The model is implicitly
  included from the JaxModel object when exporting with Neptune.

  Attributes:
    preprocessors: Sequence of processors to be executed before the model.
    postprocessors: Sequence of processors to be executed after the model.
    options: Optional processor options to configure the processing pipeline.

  Example:

    main_lib.save(
        model=jax_model,
        export_path=export_path,
        serving_configs=[
            serving_config.ServingConfig(
                serving_signature='serving_default',
                method_key='predict_fn',
                processing=processors.SequentialProcessing(
                    preprocessors=[
                        processors.TfProcessor(
                            name='tokenizer',
                            fn=tokenizer_fn,
                        ),
                        processors.TfProcessor(
                            name='fetcher',
                            fn=fetcher_fn,
                        ),
                    ],
                    postprocessors=[
                        processors.TfProcessor(
                            name='decoder',
                            fn=decoder_fn,
                        ),
                    ],
                    options=processors.ProcessingOptions(
                        tf_trackable_resources=[var1, var2]
                    ),
                )
            )
        ],
    )
  """

  preprocessors: Sequence[Processor] = dataclasses.field(default_factory=list)
  postprocessors: Sequence[Processor] = dataclasses.field(default_factory=list)
  options: ProcessingOptions = dataclasses.field(
      default_factory=ProcessingOptions
  )
