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

#include "data/cpu_embedding/python/array_caster.h"  // from @xla_serving
#include "data/kernels/cpu_embedding/cpu_embedding_resource.h"  // from @xla_serving
#include "data/kernels/cpu_embedding/cpu_embedding_serializer.h"  // from @xla_serving
#include "include/pybind11/pybind11.h"  // from @pybind11_bazel
#include "include/pybind11/stl.h"  // from @pybind11_bazel
#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil

namespace py = pybind11;

PYBIND11_MODULE(cpu_embedding_serializer_pybind, m) {
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

  m.def("serialize", &::xla_serving::data::composite::SerializeCpuEmbedding);
  m.attr("MANIFEST_ASSET_NAME") =
      py::cast(::xla_serving::data::composite::CpuEmbeddingEngineResource::
                   kManifestAssetName);
}
