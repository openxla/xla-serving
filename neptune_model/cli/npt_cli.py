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

"""Command line interface for Neptune Model.

npt-cli is a command line interface for Neptune Model. It allows users to
inspect an exported Neptune Model.
"""

from collections.abc import Sequence
import sys

from absl import app
import rich
import typer

from neptune_model.cli import constants
from neptune_model.cli import recompile
from neptune_model.cli import show

cli = typer.Typer(pretty_exceptions_enable=False)
cli.command(name='show')(show.show)
cli.command(name='recompile')(recompile.recompile)


@cli.command()
def version() -> None:
  """Prints the NPT CLI version."""
  rich.print(f'NPT CLI version: {constants.CLI_VERSION}')


def main(unused_argv: Sequence[str]) -> None:
  cli()


def run() -> None:
  """Runs the NPT CLI."""
  app.run(main, argv=sys.argv[:1])


if __name__ == '__main__':
  run()
