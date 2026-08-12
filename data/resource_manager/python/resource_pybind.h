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

#ifndef THIRD_PARTY_NEPTUNE_DATA_RESOURCE_MANAGER_PYTHON_RESOURCE_PYBIND_H_
#define THIRD_PARTY_NEPTUNE_DATA_RESOURCE_MANAGER_PYTHON_RESOURCE_PYBIND_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "data/resource_manager/cc/asset.h"  // from @neptune
#include "data/resource_manager/cc/resource.h"  // from @neptune
#include "include/pybind11/pybind11.h"  // from @pybind11_bazel
#include "include/pybind11/stl.h"  // from @pybind11_bazel
#include "include/pybind11/trampoline_self_life_support.h"  // from @pybind11_bazel
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil

namespace neptune::data {

// Trampoline class for Resource to allow overriding virtual methods in Python.
class PyResource : public Resource,
                   public pybind11::trampoline_self_life_support {
 public:
  using Resource::Resource;  // Inherit constructors
  absl::Status LoadImpl() override {
    PYBIND11_OVERRIDE_PURE(absl::Status, Resource, LoadImpl);
  }
};

// Register a specific class in the pybind11 module.
// Note: This needs to be instantiated per-class.
template <typename T>
void RegisterPybindResourceExt(pybind11::module& m, const std::string& name) {
  auto cls = pybind11::classh<T, Resource>(m, name.c_str());
  cls.def(pybind11::init<>());
  cls.def(pybind11::init<std::vector<Asset>>());
  cls.def("load", &Resource::Load);
  cls.def_property_readonly("name", &Resource::name);
  cls.def_property_readonly("assets", &Resource::assets);
  cls.def_property_readonly("class_name", &Resource::class_name);
}

}  // namespace neptune::data

#endif  // THIRD_PARTY_NEPTUNE_DATA_RESOURCE_MANAGER_PYTHON_RESOURCE_PYBIND_H_
