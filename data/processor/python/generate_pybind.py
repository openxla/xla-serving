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

# pylint: disable=line-too-long
r"""Script to parse C++ source files and auto-generate Pybind11 registration code.

Usage:
```python
  generate_pybind.py <output_file> <module_name> <src1> <src2> ...

Example:
  generate_pybind.py foo_bindings.cc foo_bindings processor.h foo.h
```
"""

import re
import sys


# Remove C/C++ comments to prevent braces inside them from confusing the
# parser.
def strip_comments(text):
  """Removes C/C++ comments from the given text."""
  text = re.sub(r'//.*', '', text)
  text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
  return text


def split_cpp_params(params_str: str) -> list[str]:
  """Splits top-level C++ function parameters."""
  params = []
  current_param = []
  depth = 0
  for char in params_str:
    if char in ('<', '('):
      depth += 1
      current_param.append(char)
    elif char in ('>', ')'):
      depth -= 1
      current_param.append(char)
    elif char == ',' and depth == 0:
      params.append(''.join(current_param).strip())
      current_param = []
    else:
      current_param.append(char)
  if current_param:
    params.append(''.join(current_param).strip())
  return [p for p in params if p]


CPP_TO_PY_TYPE = {
    'int': 'int',
    'size_t': 'int',
    'uint32_t': 'int',
    'uint64_t': 'int',
    'int32_t': 'int',
    'int64_t': 'int',
    'float': 'float',
    'double': 'float',
    'bool': 'bool',
    'string': 'str',
    'std::string': 'str',
    'jsv::HostBuffer': 'composite_processor_pybind.HostBuffer',
    'HostBuffer': 'composite_processor_pybind.HostBuffer',
}


def map_primitive(t: str) -> str:
  t = t.strip()
  return CPP_TO_PY_TYPE.get(t, 'Any')


def handle_vector(args: list[str]) -> str:
  inner = args[0] if args else 'Any'
  return f'list[{inner}]'


def handle_map(args: list[str]) -> str:
  if len(args) == 2:
    return f'dict[{args[0]}, {args[1]}]'
  return 'dict'


def handle_set(args: list[str]) -> str:
  inner = args[0] if args else 'Any'
  return f'set[{inner}]'


def handle_tuple(args: list[str]) -> str:
  return f'tuple[{", ".join(args)}]'


TEMPLATE_MAPPERS = {
    'vector': handle_vector,
    'map': handle_map,
    'set': handle_set,
    'tuple': handle_tuple,
}


def map_cpp_type_to_py(cpp_type_str: str) -> str:
  """Maps C++ types to Python type hints using a stack to handle templates."""
  cpp_type_str = cpp_type_str.strip()
  cpp_type_str = re.sub(r'^const\s+', '', cpp_type_str)
  cpp_type_str = re.sub(r'[\s&*]$', '', cpp_type_str)
  cpp_type_str = cpp_type_str.strip()

  if not cpp_type_str:
    return 'Any'

  stack = [('', [])]
  current_token = ''

  for char in cpp_type_str:
    if char.isalnum() or char in ('_', ':', '.'):
      current_token += char
    elif char == '<':
      stack.append((current_token.strip(), []))
      current_token = ''
    elif char == '>':
      if current_token.strip():
        stack[-1][1].append(map_primitive(current_token.strip()))
        current_token = ''

      base, args = stack.pop()

      py_type = 'Any'
      matched = False
      for key, mapper in TEMPLATE_MAPPERS.items():
        if key in base:
          py_type = mapper(args)
          matched = True
          break

      if not matched:
        py_type = map_primitive(base)

      stack[-1][1].append(py_type)

    elif char == ',':
      if current_token.strip():
        stack[-1][1].append(map_primitive(current_token.strip()))
        current_token = ''
    elif char.isspace():
      continue

  if current_token.strip():
    stack[-1][1].append(map_primitive(current_token.strip()))

  if stack[0][1]:
    return stack[0][1][0]

  return 'Any'


def parse_signature(params_str):
  """Parses a C++ function signature and returns a Python signature."""
  if not params_str.strip():
    return 'self'
  mapped_params = ['self']

  for i, p in enumerate(split_cpp_params(params_str)):
    p = p.strip()
    has_default = '=' in p
    if has_default:
      p = p.split('=', 1)[0].strip()
    parts = p.split()
    if not parts:
      continue

    last_token = parts[-1].replace('&', '').replace('*', '')

    type_keywords = set(CPP_TO_PY_TYPE.keys())

    if last_token in type_keywords or not parts[:-1]:
      param_name = f'arg{i}'
      cpp_type = p
    else:
      param_name = last_token
      cpp_type = ' '.join(parts[:-1])

    py_type = map_cpp_type_to_py(cpp_type)

    if param_name == 'pass':
      param_name = 'pass_'

    if has_default:
      mapped_params.append(f'{param_name}: {py_type} = ...')
    else:
      mapped_params.append(f'{param_name}: {py_type}')

  return ', '.join(mapped_params)


def generate_pyi(output_file, registrations, all_code):
  """Generates a pyi file for the given Transform registrations."""
  with open(output_file, 'w') as out:

    def write(s):
      out.write(s + '\n')

    write('# Auto-generated Pybind11 stub file')
    write('from typing import Any, overload, Sequence')
    write('from data.processor.python import composite_processor_pybind')
    write('')

    seen_pyi_names = set()
    for type_name, py_name in registrations.items():
      if py_name in seen_pyi_names:
        continue
      seen_pyi_names.add(py_name)

      short_cls = type_name.split('::')[-1]
      cls_block = re.search(
          rf'class\s+{short_cls}\s*[:{{](.*?)}};', all_code, re.DOTALL
      )
      if cls_block:
        target_code = cls_block.group(1)
      else:
        target_code = all_code

      init_params = '*args: Any, **kwargs: Any'
      exec_params = '*args: Any, **kwargs: Any'

      create_match = re.search(
          r'Create\s*\(((?:[^()]+|\([^()]*\))*)\)', target_code
      )
      if create_match:
        init_params = parse_signature(create_match.group(1))

      ret_py_type = 'Any'
      op_match = re.search(
          r'([a-zA-Z0-9_:<>\s]+)\s+operator\s*\(\)\s*\(((?:[^()]+|\([^()]*\))*)\)',
          target_code,
      )
      if op_match:
        ret_cpp_type = op_match.group(1).strip().split()[-1]
        if 'int' in ret_cpp_type or 'size_t' in ret_cpp_type:
          ret_py_type = 'int'
        elif 'float' in ret_cpp_type or 'double' in ret_cpp_type:
          ret_py_type = 'float'
        elif 'string' in ret_cpp_type:
          ret_py_type = 'str'
        elif 'bool' in ret_cpp_type:
          ret_py_type = 'bool'
        elif ret_cpp_type == 'void':
          ret_py_type = 'None'

        exec_params = parse_signature(op_match.group(2))

      write(f'class {py_name}(composite_processor_pybind.TransformBase):')
      write(f'  def __init__({init_params}) -> None: ...')
      write('  @overload')
      write('  def execute(self, context: composite_processor_pybind.ExecuteContext) -> None: ...')
      write('  @overload')
      write(f'  def execute({exec_params}) -> {ret_py_type}: ...')
      write(f'  def __call__({exec_params}) -> {ret_py_type}: ...')


def main():
  if len(sys.argv) < 4:
    print('Usage: generate_pybind.py <output_file> <module_name> <src1> ...')
    sys.exit(1)

  output_file = sys.argv[1]
  module_name = sys.argv[2]
  srcs = sys.argv[3:]

  registrations = {}

  # Tokenizer looking for:
  # 1. String literal (to be ignored so we don't count braces inside it)
  # 2. namespace X { or namespace X::Y {
  # 3. { (generic open brace)
  # 4. } (generic close brace)
  # 5. ND_REGISTER_TRANSFORM(Type, "Name")
  token_re = re.compile(
      r'(?P<str>"([^"\\]|\\.)*")|'
      r'(?P<ns>namespace\s+(?P<ns_name>[a-zA-Z_][a-zA-Z0-9_:]*)\s*\{)|'
      r'(?P<lcb>\{)|'
      r'(?P<rcb>\})|'
      r'(?P<reg>(?:ND|JD)_REGISTER_TRANSFORM\s*\(\s*(?P<type>[^,]+)\s*,\s*"(?P<name>[^"]+)"\s*\))|'
  )

  all_code = ''
  for src in srcs:
    try:
      with open(src, 'r') as f:
        content = strip_comments(f.read())
        all_code += content + '\n'
        # Stack values: string for namespace name, or None for random C++
        # block '{}'.
        namespace_stack = []

        for match in token_re.finditer(content):
          if match.group('ns'):
            namespace_stack.append(match.group('ns_name'))
          elif match.group('lcb'):
            namespace_stack.append(None)
          elif match.group('rcb'):
            if namespace_stack:
              namespace_stack.pop()
          elif match.group('reg'):
            type_name = match.group('type').strip()
            py_name = match.group('name').strip()

            # Fully qualify the C++ type name if relative
            if not type_name.startswith('::'):
              if '::' in type_name:
                type_name = '::' + type_name
              else:
                prefix = ''
                for ns in namespace_stack:
                  if ns is not None:
                    prefix += ns + '::'
                type_name = '::' + prefix + type_name

            assert type_name not in registrations, (
                'generate_pybind.py: Warning: Duplicate registration for'
                f' {type_name} in {src}'
            )
            registrations[type_name] = py_name
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          f'generate_pybind.py: Warning: Failed to parse {src}: {e}',
          file=sys.stderr,
      )

  if output_file.endswith('.pyi'):
    generate_pyi(output_file, registrations, all_code)
    return

  with open(output_file, 'w') as out:

    def write(s):
      out.write(s + '\n')

    write('// Auto-generated Pybind11 registration file')
    write('#include "data/processor/cc/core/transform_registry.h"')
    write('#include "data/processor/python/transform_pybind.h"')
    write('#include "include/pybind11/pybind11.h"')
    write('#include "pybind11_abseil/status_casters.h"')
    write('#include "pybind11_protobuf/native_proto_caster.h"')

    for src in srcs:
      if src.endswith('.h'):
        # Note: bazel genrule executes in the execroot
        write(f'#include "{src}"')

    write('namespace py = pybind11;')
    write(f'PYBIND11_MODULE({module_name}, m) {{')
    write('  try {')
    write('    py::google::ImportStatusModule();')
    write('  } catch (py::error_already_set& e) {')
    write('    if (e.matches(PyExc_ModuleNotFoundError)) {')
    write('      // Fallback for isolated bazel tests where package redefinitions might not')
    write('      // be accessible due to a hermetic build.')
    write('      py::module_::import("pybind11_abseil.status");')
    write('    } else {')
    write('      throw;')
    write('    }')
    write('  }')
    write('  pybind11_protobuf::ImportNativeProtoCasters();')
    write('  try {')
    write('    py::module::import("xla_serving_data.processor.python.composite_processor_pybind");')
    write('  } catch (py::error_already_set& e) {')
    write('    try {')
    write('      py::module::import("data.processor.python.composite_processor_pybind");')
    write('    } catch (const py::error_already_set& e2) {')
    write('      throw std::runtime_error("Failed to import both xla_serving_data and data. Original error: " + std::string(e.what()));')
    write('    }')
    write('  }')
    py_name_to_types = {}
    for type_name, py_name in registrations.items():
      if py_name not in py_name_to_types:
        py_name_to_types[py_name] = []
      py_name_to_types[py_name].append(type_name)

    for py_name, type_list in py_name_to_types.items():
      if len(type_list) > 1:
        write(
            f'  ::xla_serving::data::composite::RegisterDynamicPybindTransformExt<{type_list[0]}>(m, "{py_name}");'
        )
      else:
        write(
            f'  ::xla_serving::data::composite::RegisterPybindTransformExt<{type_list[0]}>(m, "{py_name}");'
        )
    write('}')


if __name__ == '__main__':
  main()
