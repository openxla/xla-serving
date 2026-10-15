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

#include "runtime/libs/mlir/dump_mlir_util.h"  // from @xla_serving

#include <string>

#include "gtest/gtest.h"
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/include/mlir/Support/FileUtilities.h"  // from @llvm-project
#include "xla/tsl/lib/core/status_test_util.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/test.h"  // from @xla

namespace xla_serving {
namespace {


TEST(DumpMlirModuleTest, NoEnvPrefix) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  unsetenv("JSV_DUMP_GRAPH_PREFIX");

  std::string filepath = DumpMlirOpToFile("module", module_ref.get());
  EXPECT_EQ(filepath, "(JSV_DUMP_GRAPH_PREFIX not specified)");
}

TEST(DumpMlirModuleTest, LogInfo) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  setenv("JSV_DUMP_GRAPH_PREFIX", "-", 1);

  std::string filepath = DumpMlirOpToFile("module", module_ref.get());
  EXPECT_EQ(filepath, "(stderr; requested filename: 'module')");
}

TEST(DumpMlirModuleTest, Valid) {
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module_ref =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  setenv("JSV_DUMP_GRAPH_PREFIX", tsl::testing::TmpDir().c_str(), 1);
  std::string expected_txt_module;
  {
    llvm::raw_string_ostream os(expected_txt_module);
    module_ref->getOperation()->print(os,
                                      mlir::OpPrintingFlags().useLocalScope());
    os.flush();
  }

  std::string filepath = DumpMlirOpToFile("module", module_ref.get());
  ASSERT_NE(filepath, "(JSV_DUMP_GRAPH_PREFIX not specified)");
  ASSERT_NE(filepath, "LOG(INFO)");
  ASSERT_NE(filepath, "(unavailable)");

  tsl::Env* env = tsl::Env::Default();
  std::string file_txt_module;
  TF_ASSERT_OK(tsl::ReadFileToString(env, filepath, &file_txt_module));
  EXPECT_EQ(file_txt_module, expected_txt_module);
}

}  // namespace
}  // namespace xla_serving
