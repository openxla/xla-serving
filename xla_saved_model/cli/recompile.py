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

"""XSM CLI `recompile` command."""

from typing import Annotated, NoReturn

import rich
import typer

from xla_saved_model._src.core import aot_recompiler
from xla_saved_model._src.core import file_utils
from xla_saved_model._src.core import offline_compiler


def _bail(msg: str) -> NoReturn:
  typer.secho(f"Error: {msg}", fg=typer.colors.RED, err=True)
  raise typer.Exit(1)


def recompile(
    input_model_path: Annotated[
        str,
        typer.Option(
            "--input-model-path",
            "-i",
            help="Path to the input XLA SavedModel directory.",
        ),
    ],
    output_model_path: Annotated[
        str,
        typer.Option(
            "--output-model-path",
            "-o",
            help="Destination path for the recompiled model directory.",
        ),
    ],
    platform: Annotated[
        str,
        typer.Option(
            "--platform",
            "-p",
            help="Target platform backend (cuda, tpu, cpu).",
        ),
    ],
    target_topology_label: Annotated[
        str,
        typer.Option(
            "--target-topology-label",
            "-l",
            help="Target topology label identifier saved in the manifest.",
        ),
    ],
    serialized_topology: str | None = typer.Option(
        None,
        help=(
            "Path to file containing serialized TopologyDescription proto"
            " bytes."
        ),
    ),
    autotune_database_path: str | None = typer.Option(
        None,
        help=(
            "Optional path to CUDA autotune cache. Passing a directory enables"
            " read/write per-fusion caching; passing a file (.pb/.textproto)"
            " strictly loads read-only results."
        ),
    ),
    target_topology: str | None = typer.Option(
        None,
        help=(
            "Optional hardware topology mesh specification string (e.g."
            " '1x1x1')."
        ),
    ),
    include_stablehlo: bool = typer.Option(
        True,
        "--include-stablehlo/--strip-stablehlo",
        help="Whether to preserve StableHLO bytecode in manifest.",
    ),
    version: int | None = typer.Option(
        None,
        "--version",
        "-v",
        help="Optional explicit integer model version subdirectory.",
    ),
    auto_timestamp: bool = typer.Option(
        True,
        help=(
            "Whether to place output in timestamped subdirectory if version"
            " is omitted."
        ),
    ),
    overwrite: bool = typer.Option(
        False,
        help="Overwrite existing output directory if it exists.",
    ),
) -> None:
  """Recompiles an exported XLA SavedModel for a target hardware topology."""
  if autotune_database_path and not file_utils.exists(autotune_database_path):
    _bail(f"Autotune database path not found: {autotune_database_path}")

  serialized_topology_bytes = None
  if serialized_topology:
    if not file_utils.exists(serialized_topology):
      _bail(f"Serialized topology file not found: {serialized_topology}")
    with file_utils.open_file(serialized_topology, "rb") as f:
      serialized_topology_bytes = f.read()

  if target_topology and serialized_topology:
    _bail("Cannot specify both --target-topology and --serialized-topology.")

  target_config = offline_compiler.TargetConfig(
      target_topology_label=target_topology_label,
      platform=platform.strip().lower(),
      serialized_topology=serialized_topology_bytes,
      autotune_database_path=autotune_database_path,
      target_topology=target_topology,
  )
  try:
    out_dir = aot_recompiler.recompile(
        input_model_path=input_model_path,
        output_model_path=output_model_path,
        target_configs=target_config,
        include_stablehlo=include_stablehlo,
        version=version,
        auto_timestamp=auto_timestamp,
        overwrite=overwrite,
    )
    rich.print(f"Recompiled model saved to `{out_dir}`")
  except (
      ValueError,
      FileNotFoundError,
      FileExistsError,
      RuntimeError,
  ) as e:
    _bail(str(e))
