#include "data/kernels/cpu_embedding/cpu_embedding_resource.h"  // from @neptune
#include "data/kernels/cpu_embedding/cpu_embedding_serializer.h"  // from @neptune
#include "data/cpu_embedding/python/array_caster.h"  // from @neptune
#include "include/pybind11/pybind11.h"  // from @pybind11_bazel
#include "include/pybind11/stl.h"  // from @pybind11_bazel
#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil
#include "pybind11_protobuf/native_proto_caster.h"  // from @pybind11_protobuf

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

  pybind11_protobuf::ImportNativeProtoCasters();

  m.def("serialize", &::neptune::data::composite::SerializeCpuEmbedding);
  m.attr("MANIFEST_ASSET_NAME") =
      py::cast(::neptune::data::composite::CpuEmbeddingEngineResource::
                   kManifestAssetName);
}
