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

#include <Python.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Bytecode/BytecodeWriter.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/include/mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project
#include "include/nanobind/nanobind.h"  // from @nanobind_bazel
#include "include/nanobind/stl/string.h"  // from @nanobind_bazel
#include "include/nanobind/stl/string_view.h"  // from @nanobind_bazel
#include "include/nanobind/stl/vector.h"  // from @nanobind_bazel
#include "runtime/core/stablehlo_optimizer.h"  // from @neptune
#include "xla/pjrt/mlir_to_hlo.h"  // from @xla

namespace nb = ::nanobind;

namespace {

void ThrowIfError(const absl::Status& status, absl::string_view msg,
                  absl::string_view diag_str = "") {
  if (status.ok()) return;
  throw nb::value_error(
      absl::StrCat(msg, ". Error: ", status.message(),
                   diag_str.empty()
                       ? ""
                       : absl::StrCat("\nMLIR Diagnostics:\n", diag_str))
          .c_str());
}

std::string OptimizeImpl(neptune::StablehloOptimizer* optimizer,
                         absl::string_view mlir_module_serialized,
                         const std::vector<int64_t>& variable_indices) {
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  mlir::DialectRegistry registry;
  xla::RegisterAllHloDialects(registry);
  context.appendDialectRegistry(registry);

  std::string diag_str;
  mlir::ScopedDiagnosticHandler diag_handler(
      &context, [&](mlir::Diagnostic& diag) {
        absl::StrAppend(&diag_str, diag.str(), "\n");
        for (const auto& note : diag.getNotes()) {
          absl::StrAppend(&diag_str, "  note: ", note.str(), "\n");
        }
        return mlir::success();
      });

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(mlir_module_serialized, &context);
  if (!module) {
    throw nb::value_error(
        absl::StrCat("Failed to parse MLIR module string:\n", diag_str)
            .c_str());
  }

  neptune::StablehloOptimizer::Args args;
  args.variable_indices = variable_indices;
  args.compile_options = nullptr;

  ThrowIfError(optimizer->Optimize(args, *module),
               "Failed to optimize StableHLO module", diag_str);

  std::string out_str;
  llvm::raw_string_ostream os(out_str);
  if (mlir::failed(mlir::writeBytecodeToFile(module.get(), os))) {
    throw nb::value_error("Failed to write MLIR module to bytecode.");
  }
  return out_str;
}

}  // namespace

NB_MODULE(optimizer, m) {
  nb::class_<neptune::StablehloOptimizer>(m, "StablehloOptimizer")
      .def(nb::init<>())
      .def("optimize",
           [](neptune::StablehloOptimizer* self, nb::bytes mlir_module_serialized,
              const std::vector<int64_t>& variable_indices) {
             std::string result =
                 OptimizeImpl(self,
                              absl::string_view(mlir_module_serialized.c_str(),
                                                mlir_module_serialized.size()),
                              variable_indices);
             return nb::bytes(result.data(), result.size());
           });
}
