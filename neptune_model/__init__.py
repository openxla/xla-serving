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

"""Defines exported symbols for Neptune Model."""

import importlib
import importlib.abc
import importlib.machinery
import sys


class _XlaRedirectLoader(importlib.abc.Loader):

  def __init__(self, tf_module):
    self.tf_module = tf_module

  def create_module(self, spec):
    return self.tf_module

  def exec_module(self, module):
    pass


class _XlaInterceptor(importlib.abc.MetaPathFinder):

  def find_spec(self, fullname, path, target=None):
    if fullname.startswith("neptune_model.xla.") and fullname.endswith("_pb2"):
      tf_name = fullname.replace("neptune_model.xla", "tensorflow.compiler.xla")
      try:
        tf_module = importlib.import_module(tf_name)
        return importlib.machinery.ModuleSpec(
            fullname, _XlaRedirectLoader(tf_module)
        )
      except ModuleNotFoundError:
        pass
    return None


sys.meta_path.insert(0, _XlaInterceptor())


# TODO: Add exports.

__version__ = "0.0.6"


from neptune_model._src.export.jax_model import ApplyFnBundle
from neptune_model._src.export.jax_model import JaxModel
from neptune_model._src.export.main_lib import save
from neptune_model._src.export.serving_config import ServingConfig
from neptune_model._src.processors.processor import (
    ProcessingOptions,
    SequentialProcessing,
)
from neptune_model._src.processors.tf_processor import TfProcessor
