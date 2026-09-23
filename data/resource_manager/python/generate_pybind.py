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


def main():
  if len(sys.argv) < 4:
    print('Usage: generate_pybind.py <output_file> <module_name> <src1> ...')
    sys.exit(1)

  output_file = sys.argv[1]
  module_name = sys.argv[2]
  srcs = sys.argv[3:]

  registrations = []

  # Remove C/C++ comments to prevent braces inside them from confusing the
  # parser.
  def strip_comments(text):
    text = re.sub(r'//.*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    return text

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
      r'(?P<reg>(?:ND|JD)_REGISTER_RESOURCE\s*\(\s*(?P<type>[^,]+)\s*,\s*"(?P<name>[^"]+)"\s*\))'
  )

  for src in srcs:
    try:
      with open(src, 'r') as f:
        content = strip_comments(f.read())
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

            registrations.append((type_name, py_name))
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          f'generate_pybind.py: Warning: Failed to parse {src}: {e}',
          file=sys.stderr,
      )

  with open(output_file, 'w') as out:

    def write(s):
      out.write(s + '\n')

    write('// Auto-generated Pybind11 registration file')
    write('#include "data/resource_manager/cc/resource.h"')
    write(
        '#include'
        ' "data/resource_manager/python/resource_pybind.h"'
    )
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
    write(
        '      // Fallback for isolated bazel tests where package redefinitions'
        ' might not'
    )
    write('      // be accessible due to a hermetic build.')
    write('      py::module_::import("pybind11_abseil.status");')
    write('    } else {')
    write('      throw;')
    write('    }')
    write('  }')
    write('  pybind11_protobuf::ImportNativeProtoCasters();')
    write('  try {')
    write(
        '    py::module::import("xla_serving_data.resource_manager.python.resource_manager_pybind");'
    )
    write('  } catch (py::error_already_set& e) {')
    write('    try {')
    write(
        '     '
        ' py::module::import("data.resource_manager.python.resource_manager_pybind");'
    )
    write('    } catch (const py::error_already_set& e2) {')
    write(
        '      throw std::runtime_error("Failed to import both xla_serving_data and'
        ' data. Original error: " + std::string(e.what()));'
    )
    write('    }')
    write('  }')

    for type_name, py_name in registrations:
      write(
          f'  ::xla_serving::data::RegisterPybindResourceExt<{type_name}>(m,'
          f' "{py_name}");'
      )

    write('}')


if __name__ == '__main__':
  main()
