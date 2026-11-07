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
from data.processor.python import generate_pybind


class GeneratePybindTest(absltest.TestCase):

  def test_map_cpp_type_to_py_primitive(self):
    self.assertEqual(generate_pybind.map_cpp_type_to_py("int"), "int")
    self.assertEqual(generate_pybind.map_cpp_type_to_py("float"), "float")
    self.assertEqual(generate_pybind.map_cpp_type_to_py("std::string"), "str")
    self.assertEqual(generate_pybind.map_cpp_type_to_py("bool"), "bool")
    self.assertEqual(generate_pybind.map_cpp_type_to_py("double"), "float")
    self.assertEqual(generate_pybind.map_cpp_type_to_py("unmapped_type"), "Any")

  def test_map_cpp_type_to_py_ignores_runtime_namespace(self):
    # The runtime's C++ namespace is rewritten on export, so both spellings
    # have to resolve to the same Python type.
    expected = "composite_processor_pybind.HostBuffer"
    for cpp_type in (
        "HostBuffer",
        "jsv::HostBuffer",
        "xla_serving::HostBuffer",
    ):
      self.assertEqual(generate_pybind.map_cpp_type_to_py(cpp_type), expected)

  def test_parse_signature_unnamed_namespaced_param(self):
    # An unnamed parameter must be recognised as a type under either
    # namespace, rather than being mistaken for a parameter name.
    for namespace in ("jsv", "xla_serving"):
      self.assertEqual(
          generate_pybind.parse_signature(f"{namespace}::HostBuffer"),
          "self, arg0: composite_processor_pybind.HostBuffer",
      )

  def test_map_cpp_type_to_py_vector(self):
    self.assertEqual(
        generate_pybind.map_cpp_type_to_py("std::vector<int>"), "list[int]"
    )
    self.assertEqual(
        generate_pybind.map_cpp_type_to_py("std::vector<float>"), "list[float]"
    )
    self.assertEqual(
        generate_pybind.map_cpp_type_to_py("std::vector<std::string>"),
        "list[str]",
    )

  def test_map_cpp_type_to_py_map(self):
    self.assertEqual(
        generate_pybind.map_cpp_type_to_py("std::map<int, float>"),
        "dict[int, float]",
    )
    self.assertEqual(
        generate_pybind.map_cpp_type_to_py("std::map<std::string, int>"),
        "dict[str, int]",
    )
    self.assertEqual(
        generate_pybind.map_cpp_type_to_py("std::map<int, Value>"),
        "dict[int, Any]",
    )

  def test_map_cpp_type_to_py_nested(self):
    self.assertEqual(
        generate_pybind.map_cpp_type_to_py(
            "std::vector<std::map<int, std::string>>"
        ),
        "list[dict[int, str]]",
    )
    self.assertEqual(
        generate_pybind.map_cpp_type_to_py("std::optional<int64_t>"),
        "int | None",
    )
    self.assertEqual(
        generate_pybind.map_cpp_type_to_py(
            "std::vector<std::vector<std::optional<int64_t>>>"
        ),
        "list[list[int | None]]",
    )

  def test_map_cpp_type_to_py_fallback(self):
    self.assertEqual(
        generate_pybind.map_cpp_type_to_py("std::shared_ptr<int>"),
        "Any",
    )

  def test_split_cpp_params_simple(self):
    self.assertEqual(
        generate_pybind.split_cpp_params("int a, float b"), ["int a", "float b"]
    )

  def test_split_cpp_params_template(self):
    self.assertEqual(
        generate_pybind.map_cpp_type_to_py(
            "std::vector<std::map<int, std::string>>"
        ),
        "list[dict[int, str]]",
    )

  def test_split_cpp_params_nested(self):
    self.assertEqual(
        generate_pybind.split_cpp_params(
            "std::map<int, std::vector<float>> a, bool b"
        ),
        ["std::map<int, std::vector<float>> a", "bool b"],
    )

  def test_split_cpp_params_with_parentheses(self):
    self.assertEqual(
        generate_pybind.split_cpp_params(
            "int a = std::max(1, 2), float b = 0.0f"
        ),
        ["int a = std::max(1, 2)", "float b = 0.0f"],
    )
    self.assertEqual(
        generate_pybind.split_cpp_params(
            "void (*callback)(int, float), bool flag"
        ),
        ["void (*callback)(int, float)", "bool flag"],
    )
    self.assertEqual(
        generate_pybind.split_cpp_params(
            "std::function<void(int, float)> cb, int x"
        ),
        ["std::function<void(int, float)> cb", "int x"],
    )

  def test_parse_signature_with_default_args(self):
    self.assertEqual(
        generate_pybind.parse_signature("int a = 0, float b = 1.0"),
        "self, a: int = ..., b: float = ...",
    )


if __name__ == "__main__":
  absltest.main()
