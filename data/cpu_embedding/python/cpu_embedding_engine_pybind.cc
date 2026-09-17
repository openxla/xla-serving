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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/cpu_embedding/cc/array.h"  // from @xla_serving
#include "data/cpu_embedding/cc/cpu_embedding_engine.h"  // from @xla_serving
#include "data/cpu_embedding/cc/embedding_table.h"  // from @xla_serving
#include "data/cpu_embedding/cc/feature_batch_csr_input.h"  // from @xla_serving
#include "data/cpu_embedding/cc/feature_spec.h"  // from @xla_serving
#include "data/cpu_embedding/python/array_caster.h"  // from @xla_serving
#include "include/pybind11/numpy.h"  // from @pybind11_bazel
#include "include/pybind11/pybind11.h"  // from @pybind11_bazel
#include "include/pybind11/stl.h"  // from @pybind11_bazel
#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil

namespace py = pybind11;
namespace cpu_embedding = ::neptune::data::cpu_embedding;

PYBIND11_MODULE(cpu_embedding_engine_pybind, m) {
  m.doc() = "Python extension module to access CPU embedding engine.";

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

  py::enum_<cpu_embedding::DType>(m, "DType")
      .value("UNKNOWN", cpu_embedding::DType::kUnknown)
      .value("FLOAT32", cpu_embedding::DType::kFloat32)
      .value("BFLOAT16", cpu_embedding::DType::kBfloat16)
      .value("INT32", cpu_embedding::DType::kInt32)
      .value("INT64", cpu_embedding::DType::kInt64)
      .export_values();

  // Bind Array as a class so metadata is accessible, but use the type_caster
  // for implicit conversions in function signatures.
  py::classh<cpu_embedding::Array>(m, "Array")
      .def(py::init(
          [](py::handle src) { return cpu_embedding::ArrayFromNumpy(src); }))
      .def("to_numpy",
           [](const cpu_embedding::Array& self) {
             return cpu_embedding::ArrayToNumpy(self);
           })
      .def_property_readonly("dtype", &cpu_embedding::Array::GetDType)
      .def_property_readonly("shape",
                             [](const cpu_embedding::Array& self) {
                               return std::vector<int64_t>(
                                   self.GetShape().dims.begin(),
                                   self.GetShape().dims.end());
                             })
      .def("__repr__", [](const cpu_embedding::Array& self) {
        return absl::StrCat(
            "Array(shape=", absl::StrJoin(self.GetShape().dims, ","), ")");
      });

  m.def(
      "create_embedding_table",
      [](std::string name, cpu_embedding::Array data,
         std::string combine_mode = "sum")
          -> absl::StatusOr<
              std::shared_ptr<const cpu_embedding::EmbeddingTable>> {
        enum cpu_embedding::EmbeddingTable::CombineMode combine_mode_enum;
        if (combine_mode == "sum") {
          combine_mode_enum = cpu_embedding::EmbeddingTable::CombineMode::kSum;
        } else if (combine_mode == "mean") {
          combine_mode_enum = cpu_embedding::EmbeddingTable::CombineMode::kMean;
        } else if (combine_mode == "sqrtn") {
          combine_mode_enum =
              cpu_embedding::EmbeddingTable::CombineMode::kSqrtN;
        } else {
          return absl::InvalidArgumentError(
              absl::StrCat("Invalid combine mode: ", combine_mode));
        }
        return cpu_embedding::EmbeddingTable::Create(name, std::move(data),
                                                     combine_mode_enum);
      },
      py::arg("name"), py::arg("data"), py::arg("combine_mode") = "sum");

  py::classh<cpu_embedding::EmbeddingTable>(m, "EmbeddingTable")
      .def_property_readonly("name", &cpu_embedding::EmbeddingTable::Name)
      .def_property_readonly("dtype", &cpu_embedding::EmbeddingTable::GetDType)
      .def_property_readonly("vocab_size",
                             &cpu_embedding::EmbeddingTable::VocabSize)
      .def_property_readonly("embedding_dim",
                             &cpu_embedding::EmbeddingTable::EmbeddingDim)
      .def(
          "lookup",
          [](const cpu_embedding::EmbeddingTable& self,
             std::vector<int64_t> ids) -> absl::StatusOr<cpu_embedding::Array> {
            return self.Lookup(absl::MakeConstSpan(ids));
          },
          py::arg("ids"), py::call_guard<py::gil_scoped_release>());

  m.def(
      "create_feature_batch_csr_input",
      [](std::string name, std::vector<int64_t> embedding_ids,
         std::vector<int32_t> row_pointers, std::vector<float> gains = {})
          -> absl::StatusOr<cpu_embedding::FeatureBatchCsrInput> {
        return cpu_embedding::FeatureBatchCsrInput::Create(
            std::move(name), std::move(embedding_ids), std::move(row_pointers),
            std::move(gains));
      },
      py::arg("name"), py::arg("embedding_ids"), py::arg("row_pointers"),
      py::arg("gains") = std::vector<float>());

  py::classh<cpu_embedding::FeatureBatchCsrInput>(m, "FeatureBatchCsrInput")
      .def_property_readonly("name", &cpu_embedding::FeatureBatchCsrInput::Name)
      .def_property_readonly("num_samples",
                             &cpu_embedding::FeatureBatchCsrInput::NumSamples)
      .def_property_readonly(
          "embedding_ids",
          [](const cpu_embedding::FeatureBatchCsrInput& self) {
            return std::vector<int64_t>(self.EmbeddingIds().begin(),
                                        self.EmbeddingIds().end());
          })
      .def_property_readonly(
          "row_pointers",
          [](const cpu_embedding::FeatureBatchCsrInput& self) {
            return std::vector<int32_t>(self.RowPointers().begin(),
                                        self.RowPointers().end());
          })
      .def_property_readonly(
          "weights", [](const cpu_embedding::FeatureBatchCsrInput& self) {
            return std::vector<float>(self.Weights().begin(),
                                      self.Weights().end());
          });

  py::class_<cpu_embedding::CpuEmbeddingEngine::Options>(m, "Options")
      .def(py::init([](int max_cpu_slots) {
             return cpu_embedding::CpuEmbeddingEngine::Options{
                 .max_cpu_slots = max_cpu_slots};
           }),
           py::arg("max_cpu_slots") = 0)
      .def_readwrite(
          "max_cpu_slots",
          &cpu_embedding::CpuEmbeddingEngine::Options::max_cpu_slots);

  py::classh<cpu_embedding::CpuEmbeddingEngine>(m, "CpuEmbeddingEngine")
      .def(py::init<>())
      .def(py::init<cpu_embedding::CpuEmbeddingEngine::Options>(),
           py::arg("options"))
      .def("register_feature_spec",
           &cpu_embedding::CpuEmbeddingEngine::RegisterFeatureSpec,
           py::arg("feature_spec"))
      .def("add_local_table", &cpu_embedding::CpuEmbeddingEngine::AddLocalTable,
           py::arg("name"), py::arg("table"))
      .def(
          "execute",
          [](const cpu_embedding::CpuEmbeddingEngine& self,
             std::vector<cpu_embedding::FeatureBatchCsrInput> feature_batches)
              -> absl::StatusOr<std::vector<cpu_embedding::Array>> {
            return self.Execute(absl::MakeConstSpan(feature_batches));
          },
          py::arg("feature_batches"), py::call_guard<py::gil_scoped_release>())
      .def(
          "execute_to_numpy",
          [](const cpu_embedding::CpuEmbeddingEngine& self,
             std::string feature_name, std::vector<int64_t> embedding_ids,
             std::vector<int32_t> row_pointers,
             std::vector<float> weights) -> absl::StatusOr<py::array_t<float>> {
            auto buffer = std::make_unique<std::vector<float>>();
            std::vector<float>* buffer_ptr = buffer.get();

            auto result = self.Execute<float>(
                feature_name, absl::MakeConstSpan(embedding_ids),
                absl::MakeConstSpan(row_pointers), absl::MakeConstSpan(weights),
                [buffer_ptr](size_t size) {
                  buffer_ptr->resize(size);
                  return absl::MakeSpan(*buffer_ptr);
                });
            if (!result.ok()) {
              return result.status();
            }

            py::capsule free_buffer(buffer.release(), [](void* f) {
              delete reinterpret_cast<std::vector<float>*>(f);
            });

            return py::array_t<float>(
                {static_cast<ssize_t>(buffer_ptr->size())}, {sizeof(float)},
                buffer_ptr->data(), free_buffer);
          },
          py::arg("feature_name"), py::arg("embedding_ids"),
          py::arg("row_pointers"), py::arg("weights") = std::vector<float>());

  py::classh<cpu_embedding::FeatureSpec>(m, "FeatureSpec")
      .def(py::init<std::string, std::string, int32_t>(), py::arg("name"),
           py::arg("table_name"), py::arg("max_sequence_length") = 0)
      .def_readwrite("name", &cpu_embedding::FeatureSpec::name)
      .def_readwrite("table_name", &cpu_embedding::FeatureSpec::table_name)
      .def_readwrite("max_sequence_length",
                     &cpu_embedding::FeatureSpec::max_sequence_length);
}
