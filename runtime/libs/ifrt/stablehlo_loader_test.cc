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

#include "runtime/libs/ifrt/stablehlo_loader.h"  // from @xla_serving

#include <string>

#include "xla/tsl/platform/resource_loader.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla

namespace xla_serving {
namespace {
absl::StatusOr<std::string> GetStableHloString(absl::string_view mlir_path) {
  std::string content;
  TF_RETURN_IF_ERROR(tsl::ReadFileToString(
      tsl::Env::Default(), std::string(mlir_path), &content));
  return content;
}

TEST(StableHloLoaderTest, Load) {
  const std::string kMlirPath = tsl::GetDataDependencyFilepath(
      "runtime/libs/ifrt/testdata/simple.serializedmlir");

  TF_ASSERT_OK_AND_ASSIGN(std::string stablehlo_str,
                          GetStableHloString(kMlirPath));

  StableHloLoader loader;
  auto module = loader.Load(stablehlo_str).value();
  ASSERT_TRUE(module);
  ASSERT_EQ(module->getName()->str(), "jit_jax_model_fn");
}
}  // namespace
}  // namespace xla_serving
