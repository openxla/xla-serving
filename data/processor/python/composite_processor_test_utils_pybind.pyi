import numpy as np

from .third_party.neptune.data.processor.python import composite_processor_pybind

def is_numpy_eigen_aligned(arr: np.ndarray) -> bool: ...
def get_ref_count_from_ref(
    hb: composite_processor_pybind.HostBuffer,
) -> int: ...
def run_transform_methods_concurrently(
    transform: composite_processor_pybind.TransformBase,
) -> None: ...

