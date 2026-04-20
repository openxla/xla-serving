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

from absl.testing import absltest
from absl.testing import parameterized

from .net.proto2.contrib.pyutil import compare
from neptune_model._src.core import orchestration_proto_utils
from neptune_model._src.core import shlo_type


class OrchestrationProtoUtilsTest(compare.Proto2Assertions, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='with_dict',
          signature={
              'a': shlo_type.ShloTensorSpec([1], shlo_type.ShloDType.f32),
              'b': shlo_type.ShloTensorSpec([2], shlo_type.ShloDType.i32),
          },
          expected=[
              ('a', shlo_type.ShloTensorSpec([1], shlo_type.ShloDType.f32)),
              ('b', shlo_type.ShloTensorSpec([2], shlo_type.ShloDType.i32)),
          ],
          error=None,
      ),
      dict(
          testcase_name='with_spec_name',
          signature=[
              shlo_type.ShloTensorSpec([1], shlo_type.ShloDType.f32, name='a'),
              shlo_type.ShloTensorSpec([2], shlo_type.ShloDType.i32, name='b'),
          ],
          expected=[
              (
                  'a',
                  shlo_type.ShloTensorSpec(
                      [1], shlo_type.ShloDType.f32, name='a'
                  ),
              ),
              (
                  'b',
                  shlo_type.ShloTensorSpec(
                      [2], shlo_type.ShloDType.i32, name='b'
                  ),
              ),
          ],
          error=None,
      ),
      dict(
          testcase_name='mixed',
          signature={
              'x': shlo_type.ShloTensorSpec(
                  [1], shlo_type.ShloDType.f32, name='a'
              ),
              'y': shlo_type.ShloTensorSpec([2], shlo_type.ShloDType.i32),
          },
          expected=[
              (
                  'a',
                  shlo_type.ShloTensorSpec(
                      [1], shlo_type.ShloDType.f32, name='a'
                  ),
              ),
              ('y', shlo_type.ShloTensorSpec([2], shlo_type.ShloDType.i32)),
          ],
          error=None,
      ),
      dict(
          testcase_name='missing_name',
          signature=[shlo_type.ShloTensorSpec([1], shlo_type.ShloDType.f32)],
          expected=None,
          error=ValueError,
      ),
      dict(
          testcase_name='duplicate_name',
          signature=[
              shlo_type.ShloTensorSpec([1], shlo_type.ShloDType.f32, name='a'),
              shlo_type.ShloTensorSpec([2], shlo_type.ShloDType.i32, name='a'),
          ],
          expected=None,
          error=ValueError,
      ),
  )
  def test_calculate_named_signature(self, signature, expected, error):
    if error:
      with self.assertRaises(error):
        orchestration_proto_utils._calculate_named_signature(signature)
    else:
      self.assertEqual(
          orchestration_proto_utils._calculate_named_signature(signature), expected
      )

  def test_to_named_signature_proto(self):
    proto = orchestration_proto_utils.to_named_signature_proto(
        shlo_type.ShloTensorSpec([1], shlo_type.ShloDType.f32, name='a'),
        shlo_type.ShloTensorSpec([2], shlo_type.ShloDType.i32, name='b')
    )
    expected_proto = """
      inputs {
        named_tensor_types {
          name: "a"
          tensor_type {
            shape { shape_with_known_rank { dimension_sizes { size: 1 } } }
            dtype: f32
          }
        }
      }
      outputs {
        named_tensor_types {
          name: "b"
          tensor_type {
            shape { shape_with_known_rank { dimension_sizes { size: 2 } } }
            dtype: si32
          }
        }
      }
    """
    self.assertProto2Equal(expected_proto, proto)

  def test_to_named_signature_proto_fail(self):
    with self.assertRaisesRegex(ValueError, 'must have not None names'):
      orchestration_proto_utils.to_named_signature_proto(
          [shlo_type.ShloTensorSpec([1], shlo_type.ShloDType.f32)],
          shlo_type.ShloTensorSpec([2], shlo_type.ShloDType.i32, name='b'),
      )


if __name__ == '__main__':
  absltest.main()