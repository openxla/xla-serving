// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <thread>
#include <vector>

#include "data/processor/cc/core/transform.h"  // from @neptune
#include "data/processor/cc/utils/buffer_utils.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune
#include "include/pybind11/gil.h"  // from @pybind11_bazel
#include "include/pybind11/numpy.h"  // from @pybind11_bazel
#include "include/pybind11/pybind11.h"  // from @pybind11_bazel
#include "include/pybind11/pytypes.h"  // from @pybind11_bazel
#include "include/pybind11/stl.h"  // from @pybind11_bazel
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil
#include "pybind11_protobuf/native_proto_caster.h"  // from @pybind11_protobuf

namespace neptune::data::composite {
namespace {

namespace py = pybind11;

PYBIND11_MODULE(composite_processor_test_utils_pybind, m) {
  m.doc() = "Composite Processor Test Utilities Python bindings";

  try {
    py::google::ImportStatusModule();
  } catch (py::error_already_set& e) {
    if (e.matches(PyExc_ModuleNotFoundError)) {
      // Fallback for isolated bazel tests where package redefinitions might not
      // be accessible due to a hermetic build.
      py::module_::import("pybind11_abseil.status");
    } else {
      throw;
    }
  }
  pybind11_protobuf::ImportNativeProtoCasters();

  m.def("is_numpy_eigen_aligned", [](py::array ref) -> bool {
    return utils::IsBufferEigenAligned(ref.data());
  });

  m.def("get_ref_count_from_ref",
        [](const neptune::HostBuffer& hb) { return hb.ref_count(); });

  m.def("run_transform_methods_concurrently",
        [](const std::shared_ptr<TransformBase>& transform) {
          pybind11::gil_scoped_release release;
          std::vector<std::thread> threads;
          threads.reserve(5);
          for (int i = 0; i < 5; ++i) {
            threads.emplace_back([transform]() {
              for (int j = 0; j < 20; ++j) {
                auto input_types = transform->InputTypes();
                auto output_types = transform->OutputTypes();
                auto debug_string = transform->DebugString();

                if (!input_types.empty()) {
                  (void)input_types[0];
                }
                if (!output_types.empty()) {
                  (void)output_types[0];
                }
                (void)debug_string.length();
              }
            });
          }
          for (auto& thread : threads) {
            thread.join();
          }
        });
}

}  // namespace
}  // namespace neptune::data::composite
