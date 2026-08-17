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

"""Test utils for Neptune data library."""

import collections.abc

from google.protobuf import json_format
from google.protobuf import message

MessageToDict = json_format.MessageToDict


def assert_proto2_equal(
    test_case,
    actual: message.Message,
    expected: message.Message,
    ignore_repeated_fields_order: bool = False,
):
  """Asserts that two protos are equal."""

  def _normalize(obj):
    if isinstance(obj, collections.abc.Mapping):
      return tuple(sorted((k, _normalize(v)) for k, v in obj.items()))
    elif isinstance(obj, collections.abc.Iterable) and not isinstance(obj, str):
      return tuple(sorted(_normalize(x) for x in obj))
    else:
      return obj

  dict_actual = MessageToDict(actual, preserving_proto_field_name=True)
  dict_expected = MessageToDict(expected, preserving_proto_field_name=True)

  if ignore_repeated_fields_order:
    test_case.assertEqual(_normalize(dict_actual), _normalize(dict_expected))
  else:
    test_case.assertEqual(dict_actual, dict_expected)
