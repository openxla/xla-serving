#include "third_party/neptune/runtime/libs/mlir/dump_mlir_util.h"

#include <string>

#include "testing/base/public/gunit.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/Support/raw_ostream.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinOps.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/Location.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/MLIRContext.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/OperationSupport.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/OwningOpRef.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Pass/PassManager.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Support/FileUtilities.h"
#include "third_party/tensorflow/compiler/xla/tsl/lib/core/status_test_util.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/env.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/test.h"

namespace jsv {
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
}  // namespace jsv
