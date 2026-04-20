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

"""Constants for Neptune Export."""

ORCHESTRATION_SUPPLEMENTAL_NAME: str = "orchestration"

DEFAULT_ORCHESTRATION_SUPPLEMENTAL_FILENAME: str = "orchestration.pb"

DEFAULT_CHECKPOINT_SUBDIR: str = "checkpoint"

WEIGHTS_NAME: str = "weights"

ORBAX_CHECKPOINT_MIME_TYPE: str = "application/x.orbax-checkpoint"


################################################################################
# Logging Messages
################################################################################
# Prefix to use for prepending to logging messages
LOG_PREFIX: str = "[NeptuneExport]"
