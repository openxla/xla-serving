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

import numpy as np

from data.processor.python import composite_processor_pybind

def is_numpy_eigen_aligned(arr: np.ndarray) -> bool: ...
def get_ref_count_from_ref(
    hb: composite_processor_pybind.HostBuffer,
) -> int: ...
def run_transform_methods_concurrently(
    transform: composite_processor_pybind.TransformBase,
) -> None: ...

