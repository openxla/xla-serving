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

"""XLA Serving Data."""

import os
import sys

# The wheel bundles XLA/TSL statically into libxla_serving_data_core.so. A
# process that also imports another wheel bundling its own XLA/TSL -- most
# commonly TensorFlow, via libtensorflow_framework.so -- ends up with two
# copies of those symbols. Whichever library is loaded first wins the global
# lookup, so our static initializers can register into the other copy's
# registry under a layout they do not agree on. The observed failure is a
# segfault in tsl::AllocatorFactoryRegistry::Register while dlopen() runs the
# initializers of any extension module in this package, but only when
# TensorFlow was imported first.
#
# RTLD_DEEPBIND makes the loader search a library's own scope before the
# global scope, so each copy consistently uses its own symbols.
#
# This deliberately leaves the flag set for the rest of the process rather
# than restoring it. The extension modules in this package are imported
# lazily by submodules, and each one needs DEEPBIND at its own dlopen(); a
# narrow window around a preload of the core library here is not enough, and
# still segfaults. Eagerly importing every extension module just to close the
# window would be worse. The flag only changes symbol resolution order for
# self-contained libraries, which is the norm for wheels shipping bundled
# native code.
#
# DEEPBIND is glibc-only. On platforms without it there is nothing to do:
# those builds do not bundle a conflicting XLA/TSL copy.
if sys.platform.startswith("linux") and hasattr(os, "RTLD_DEEPBIND"):
  sys.setdlopenflags(sys.getdlopenflags() | os.RTLD_DEEPBIND)
