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
