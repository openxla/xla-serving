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

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "data/resource_manager/cc/asset.h"  // from @xla_serving
#include "data/resource_manager/cc/resource.h"  // from @xla_serving
#include "data/resource_manager/cc/resource_manager.h"  // from @xla_serving
#include "data/resource_manager/proto/resource_manager.pb.h"  // from @xla_serving
#include "data/resource_manager/python/resource_pybind.h"  // from @xla_serving
#include "include/pybind11/pybind11.h"  // from @pybind11_bazel
#include "include/pybind11/stl.h"  // from @pybind11_bazel
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil
#include "pybind11_protobuf/native_proto_caster.h"  // from @pybind11_protobuf

namespace xla_serving::data {
namespace {

namespace py = pybind11;

PYBIND11_MODULE(resource_manager_pybind, m) {
  try {
    pybind11::google::ImportStatusModule();
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

  py::class_<Asset>(m, "CppAsset")
      .def(py::init([](absl::string_view name, absl::string_view path) {
        return Asset{.name = std::string(name), .path = std::string(path)};
      }))
      .def(py::init([](absl::string_view name, absl::string_view path,
                       absl::string_view content) {
        return Asset{.name = std::string(name),
                     .path = std::string(path),
                     .content = std::string(content)};
      }))
      .def("__repr__", [](const Asset& asset) { return absl::StrCat(asset); })
      .def_readonly("name", &Asset::name)
      .def_readonly("path", &Asset::path)
      .def_readonly("content", &Asset::content)
      .def("load_content", &Asset::LoadContent);

  py::classh<Resource, PyResource>(m, "CppResource")
      .def(py::init<>())
      .def(py::init<std::vector<Asset>>())
      .def(py::init<std::string, std::vector<Asset>, std::string>())
      .def(py::init<uint64_t, std::string, std::vector<Asset>, std::string>())
      .def("load", &Resource::Load)
      .def("__repr__",
           [](const Resource& resource) { return absl::StrCat(resource); })
      .def_property_readonly("name", &Resource::name)
      .def_property_readonly("id", &Resource::id)
      .def_property_readonly("assets", &Resource::assets)
      .def_property_readonly("class_name", &Resource::class_name);

  py::class_<ResourceManager>(m, "CppResourceManager")
      .def(py::init<>())
      .def("add_resource", &ResourceManager::AddResource)
      .def("add_resource_ignoring_duplicates",
           &ResourceManager::AddResourceIgnoringDuplicates)
      .def("get_resource", &ResourceManager::GetResource)
      .def("get_all_assets", &ResourceManager::GetAllAssets)
      .def("get_all_resources", &ResourceManager::GetAllResources)
      .def("to_proto", &ResourceManager::ToProto)
      .def_static("from_proto", &ResourceManager::FromProto)
      .def_static("create", &ResourceManager::Create);
}

}  // namespace
}  // namespace xla_serving::data
