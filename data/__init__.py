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

"""Neptune Data package."""

import os
import sys

# In OSS, libneptune_data_core.so bundles internal XLA/TSL symbols (e.g.
# AllocatorFactory registry). When loaded alongside other packages that also
# bundle XLA/TSL (such as TensorFlow), RTLD_DEEPBIND ensures that the library
# own symbol definitions are preferred, preventing duplicate registration
# conflicts (e.g. DefaultCPUAllocator collisions).
if sys.platform.startswith("linux"):
  _flags = os.RTLD_LOCAL | os.RTLD_NOW | getattr(os, "RTLD_DEEPBIND", 0x8)
  sys.setdlopenflags(_flags)
