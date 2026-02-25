#include "third_party/neptune/runtime/libs/ifrt/stablehlo_loader.h"

#include <string>

#include "devtools/build/runtime/get_runfiles_dir.h"
#include "file/base/helpers.h"
#include "file/base/options.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinOps.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/OwningOpRef.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"

namespace jsv {
namespace {
absl::StatusOr<std::string> GetStableHloString(absl::string_view mlir_path) {
  return file::GetContents(mlir_path, file::Defaults());
}

TEST(StableHloLoaderTest, Load) {
  const std::string kMlirPath = devtools_build::GetDataDependencyFilepath(
      "google3/third_party/neptune/runtime/libs/ifrt/testdata/simple.serializedmlir");

  TF_ASSERT_OK_AND_ASSIGN(std::string stablehlo_str,
                          GetStableHloString(kMlirPath));

  StableHloLoader loader;
  auto module = loader.Load(stablehlo_str).value();
  ASSERT_TRUE(module);
  ASSERT_EQ(module->getName()->str(), "jit_jax_model_fn");
}
}  // namespace
}  // namespace jsv
