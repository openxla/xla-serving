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

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "xla/tsl/platform/resource_loader.h"
#include "google/protobuf/any.pb.h"
#include "google/protobuf/wrappers.pb.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "runtime/libs/ifrt/executable_build_utils.h"  // from @xla_serving
#include "runtime/libs/ifrt/stablehlo_loader.h"  // from @xla_serving
#include "google/protobuf/message.h"
#include "xla/pjrt/pjrt_executable.h"  // from @xla
#include "xla/pjrt/proto/compile_options.pb.h"  // from @xla
#include "xla/python/ifrt/compiler.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/executable.h"  // from @xla
#include "xla/python/ifrt/mock.h"  // from @xla
#include "xla/python/ifrt/program.h"  // from @xla
#include "xla/python/ifrt/ref_wrapper.h"  // from @xla
#include "xla/python/ifrt/rtti.h"  // from @xla
#include "xla/python/pjrt_ifrt/xla_compiler.h"  // from @xla
#include "xla/service/compilation_environments.h"  // from @xla
#include "xla/service/device_assignment.h"  // from @xla
#include "xla/tsl/concurrency/ref_count.h"  // from @xla

namespace xla_serving {
namespace {

using ::testing::_;
using ::testing::Return;

class BuildIfrtExecutableTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    // Register a process new env fn for google::protobuf::StringValue so that
    // the CompilationEnvironments proto can be deserialized.
    // This is a hack for testing purposes. This only needs to be done once
    // for the entire test suite.
    xla::CompilationEnvironments::RegisterProcessNewEnvFn(
        google::protobuf::StringValue::descriptor(),
        [](std::unique_ptr<tsl::protobuf::Message> env)
            -> absl::StatusOr<std::unique_ptr<tsl::protobuf::Message>> {
          return env;
        });
  }

  void SetUp() override {
    mock_client_ = std::make_shared<xla::ifrt::MockClient>();
    mock_compiler_ = std::make_shared<xla::ifrt::MockCompiler>();
    mock_device_ = std::make_shared<xla::ifrt::MockDevice>();
    mock_device_list_ = tsl::MakeRef<xla::ifrt::MockDeviceList>();
    mock_device_list_ref_ =
        xla::ifrt::RCReferenceWrapper<xla::ifrt::DeviceList>(mock_device_list_);
    std::vector<xla::ifrt::Device*> devices = {mock_device_.get()};
    ON_CALL(*mock_device_list_, devices())
        .WillByDefault(Return(absl::MakeConstSpan(devices)));
    ON_CALL(*mock_device_list_, AddressableDeviceList())
        .WillByDefault(Return(mock_device_list_.get()));

    ON_CALL(*mock_client_, GetDefaultCompiler())
        .WillByDefault(Return(mock_compiler_.get()));

    xla::DeviceAssignment default_da(1, 1);
    default_da(0, 0) = 0;
    ON_CALL(*mock_client_, GetDefaultDeviceAssignment(1, 1))
        .WillByDefault(Return(default_da));
    ON_CALL(*mock_client_, LookupDevice(xla::ifrt::DeviceId(0)))
        .WillByDefault(Return(mock_device_.get()));
    ON_CALL(*mock_client_,
                MakeDeviceList(testing::ElementsAre(mock_device_.get())))
        .WillByDefault(Return(mock_device_list_ref_));
  }

  std::shared_ptr<xla::ifrt::MockClient> mock_client_;
  std::shared_ptr<xla::ifrt::MockCompiler> mock_compiler_;
  std::shared_ptr<xla::ifrt::MockDevice> mock_device_;
  tsl::RCReference<xla::ifrt::MockDeviceList> mock_device_list_;
  xla::ifrt::DeviceListRef mock_device_list_ref_;
};

TEST_F(BuildIfrtExecutableTest, CompEnvsStrippedOnGce) {
  const std::string kMlirPath = tsl::GetDataDependencyFilepath(
      "runtime/libs/ifrt/testdata/"
      "simple.mlir");

  StableHloLoader loader;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      loader.LoadFromFile(kMlirPath).value();
  ASSERT_TRUE(module);

  BuildIfrtExecutableOptions options;
  options.num_replicas = 1;
  options.num_cores_per_replica = 1;
  options.device_assignment = *mock_client_->GetDefaultDeviceAssignment(1, 1);
  google::protobuf::StringValue test_env_value;
  test_env_value.set_value("test_value");
  google::protobuf::Any test_env_any;
  test_env_any.PackFrom(test_env_value);
  auto xla_compile_options = std::make_unique<xla::CompileOptionsProto>();
  options.compile_options = xla_compile_options.get();
  options.compile_options->mutable_executable_build_options()
      ->mutable_comp_envs()
      ->mutable_environments()
      ->Add(std::move(test_env_any));

  std::unique_ptr<xla::ifrt::CompileOptions> captured_ifrt_options;
  EXPECT_CALL(*mock_compiler_, CompileAndLoad(_, _))
      .WillOnce([&](std::unique_ptr<xla::ifrt::Program> program,
                    std::unique_ptr<xla::ifrt::CompileOptions> ifrt_options) {
        captured_ifrt_options = std::move(ifrt_options);
        return xla::ifrt::LoadedExecutableRef(
            std::make_unique<xla::ifrt::MockLoadedExecutable>());
      });

  ASSERT_OK(BuildIfrtExecutable(module.get(), options, mock_client_,
                                /*persistent_compilation_cache=*/nullptr));
  ASSERT_NE(captured_ifrt_options, nullptr);
  auto* xla_ifrt_options = xla::ifrt::dyn_cast<xla::ifrt::XlaCompileOptions>(
      captured_ifrt_options.get());
  ASSERT_NE(xla_ifrt_options, nullptr);

  EXPECT_FALSE(xla_ifrt_options->compile_options.executable_build_options
                   .mutable_comp_envs()
                   ->HasEnv<google::protobuf::StringValue>());
}

TEST_F(BuildIfrtExecutableTest, DoesNotHaveExecutableBuildOptionsSucceeds) {
  const std::string kMlirPath = tsl::GetDataDependencyFilepath(
      "runtime/libs/ifrt/testdata/"
      "simple.mlir");

  StableHloLoader loader;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      loader.LoadFromFile(kMlirPath).value();
  ASSERT_TRUE(module);

  BuildIfrtExecutableOptions options;
  options.num_replicas = 1;
  options.num_cores_per_replica = 1;
  options.device_assignment = *mock_client_->GetDefaultDeviceAssignment(1, 1);
  google::protobuf::StringValue test_env_value;
  test_env_value.set_value("test_value");
  google::protobuf::Any test_env_any;
  test_env_any.PackFrom(test_env_value);
  auto xla_compile_options = std::make_unique<xla::CompileOptionsProto>();
  options.compile_options = xla_compile_options.get();
  options.compile_options->clear_executable_build_options();

  std::unique_ptr<xla::ifrt::CompileOptions> captured_ifrt_options;
  EXPECT_CALL(*mock_compiler_, CompileAndLoad(_, _))
      .WillOnce([&](std::unique_ptr<xla::ifrt::Program> program,
                    std::unique_ptr<xla::ifrt::CompileOptions> ifrt_options) {
        captured_ifrt_options = std::move(ifrt_options);
        return xla::ifrt::LoadedExecutableRef(
            std::make_unique<xla::ifrt::MockLoadedExecutable>());
      });

  ASSERT_OK(BuildIfrtExecutable(module.get(), options, mock_client_,
                                /*persistent_compilation_cache=*/nullptr));
  ASSERT_NE(captured_ifrt_options, nullptr);
  auto* xla_ifrt_options = xla::ifrt::dyn_cast<xla::ifrt::XlaCompileOptions>(
      captured_ifrt_options.get());
  ASSERT_NE(xla_ifrt_options, nullptr);

  EXPECT_FALSE(xla_ifrt_options->compile_options.executable_build_options
                   .mutable_comp_envs()
                   ->HasEnv<google::protobuf::StringValue>());
}

TEST_F(BuildIfrtExecutableTest, DoesNotHaveCompEnvsSucceeds) {
  const std::string kMlirPath = tsl::GetDataDependencyFilepath(
      "runtime/libs/ifrt/testdata/"
      "simple.mlir");

  StableHloLoader loader;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      loader.LoadFromFile(kMlirPath).value();
  ASSERT_TRUE(module);

  BuildIfrtExecutableOptions options;
  options.num_replicas = 1;
  options.num_cores_per_replica = 1;
  options.device_assignment = *mock_client_->GetDefaultDeviceAssignment(1, 1);
  google::protobuf::StringValue test_env_value;
  test_env_value.set_value("test_value");
  google::protobuf::Any test_env_any;
  test_env_any.PackFrom(test_env_value);
  auto xla_compile_options = std::make_unique<xla::CompileOptionsProto>();
  options.compile_options = xla_compile_options.get();
  options.compile_options->mutable_executable_build_options()
      ->clear_comp_envs();

  std::unique_ptr<xla::ifrt::CompileOptions> captured_ifrt_options;
  EXPECT_CALL(*mock_compiler_, CompileAndLoad(_, _))
      .WillOnce([&](std::unique_ptr<xla::ifrt::Program> program,
                    std::unique_ptr<xla::ifrt::CompileOptions> ifrt_options) {
        captured_ifrt_options = std::move(ifrt_options);
        return xla::ifrt::LoadedExecutableRef(
            std::make_unique<xla::ifrt::MockLoadedExecutable>());
      });

  ASSERT_OK(BuildIfrtExecutable(module.get(), options, mock_client_,
                                /*persistent_compilation_cache=*/nullptr));
  ASSERT_NE(captured_ifrt_options, nullptr);
  auto* xla_ifrt_options = xla::ifrt::dyn_cast<xla::ifrt::XlaCompileOptions>(
      captured_ifrt_options.get());
  ASSERT_NE(xla_ifrt_options, nullptr);

  EXPECT_FALSE(xla_ifrt_options->compile_options.executable_build_options
                   .mutable_comp_envs()
                   ->HasEnv<google::protobuf::StringValue>());
}

}  // namespace
}  // namespace xla_serving
