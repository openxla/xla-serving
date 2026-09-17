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

"""Required __init__ file for package."""

# pylint: disable=g-importing-member
from xla_saved_model.protos import manifest_pb2
from xla_saved_model.protos import type_pb2
from xla_saved_model._src.core.constants import *
from xla_saved_model._src.core.function import Function
from xla_saved_model._src.core.persistence_lib import GlobalSupplemental
from xla_saved_model._src.core.persistence_lib import load
from xla_saved_model._src.core.persistence_lib import save
from xla_saved_model._src.core.persistence_lib import SaveOptions
from xla_saved_model._src.core.saveable import Saveable
from xla_saved_model._src.core.serializable_function import SerializableFunction
from xla_saved_model._src.core.shlo_function import ShloFunction
from xla_saved_model._src.core.shlo_function import ShloFunctionSupplementalInfo
from xla_saved_model._src.core.shlo_type import np_dtype_to_shlo_dtype
from xla_saved_model._src.core.shlo_type import Sharding
from xla_saved_model._src.core.shlo_type import shlo_dtype_to_np_dtype
from xla_saved_model._src.core.shlo_type import ShloDimSize
from xla_saved_model._src.core.shlo_type import ShloDType
from xla_saved_model._src.core.shlo_type import ShloShape
from xla_saved_model._src.core.shlo_type import ShloTensorSpec
from xla_saved_model._src.core.type_proto_utils import from_type_proto
from xla_saved_model._src.core.type_proto_utils import to_function_signature_proto
from xla_saved_model._src.core.unstructured_data import UnstructuredData
from xla_saved_model._src.core.unstructured_data import UnstructuredDataWithExtName
from xla_saved_model._src.core.value import ExternalValue
from xla_saved_model._src.core.value import Value

# pylint: enable=g-importing-member

OpSharding = Sharding
