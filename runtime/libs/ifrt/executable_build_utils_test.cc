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

#include "runtime/libs/ifrt/executable_build_utils.h"  // from @xla_serving

#include <cstdint>
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
#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/container/flat_hash_set.h"  // from @abseil-cpp
#include "absl/functional/any_invocable.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_split.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/time/clock.h"  // from @abseil-cpp
#include "absl/time/time.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "runtime/libs/ifrt/persistent_compilation_cache/ifrt_persistent_compilation_cache.h"  // from @xla_serving
#include "runtime/libs/ifrt/stablehlo_loader.h"  // from @xla_serving
#include "xla/backends/cpu/nanort/ifrt_client.h"  // from @xla
#include "xla/pjrt/pjrt_common.h"  // from @xla
#include "xla/pjrt/pjrt_compiler.h"  // from @xla
#include "xla/pjrt/pjrt_executable.h"  // from @xla
#include "xla/pjrt/proto/compile_options.pb.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/compiler.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/executable.h"  // from @xla
#include "xla/python/ifrt/hlo/hlo_program.h"  // from @xla
#include "xla/python/ifrt/mock.h"  // from @xla
#include "xla/python/ifrt/program.h"  // from @xla
#include "xla/python/ifrt/ref_wrapper.h"  // from @xla
#include "xla/python/ifrt/rtti.h"  // from @xla
#include "xla/python/ifrt/test_util.h"  // from @xla
#include "xla/python/pjrt_ifrt/xla_compiler.h"  // from @xla
#include "xla/service/compilation_environments.h"  // from @xla
#include "xla/service/device_assignment.h"  // from @xla
#include "xla/tsl/concurrency/ref_count.h"  // from @xla
#include "xla/tsl/lib/core/status_test_util.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/util/proto/proto_matchers.h"  // from @xla

namespace xla_serving {
namespace {

using ::testing::_;
using ::testing::Return;

class FakePersistentCompilationCache : public IfrtPersistentCompilationCache {
 public:
  explicit FakePersistentCompilationCache(
      std::shared_ptr<xla::ifrt::Client> ifrt_client)
      : ifrt_client_(ifrt_client) {}

  absl::StatusOr<xla::ifrt::LoadedExecutableRef> LookupLoadedExecutableOrCreate(
      std::unique_ptr<xla::ifrt::HloProgram> hlo_program,
      xla::ifrt::DeviceListRef device_list,
      const xla::CompileOptions& xla_compile_options, xla::ifrt::Client* client,
      absl::AnyInvocable<absl::StatusOr<xla::ifrt::LoadedExecutableRef>(
          std::unique_ptr<xla::ifrt::Program> program,
          std::unique_ptr<xla::ifrt::CompileOptions> options)>
          value_fn) override {
    // For both cache hit and miss, we use CompileAndLoad to load the
    // executable. The fake cache only makes note of hits and miss counts.
    TF_ASSIGN_OR_RETURN(
        uint64_t fingerprint,
        hlo_program->Fingerprint(xla::ifrt::HloProgram::FingerprintOptions()));
    absl::MutexLock lock(mu_);
    if (cache_.contains(fingerprint)) {
      cache_hit_counter_++;
      return ifrt_client_->GetDefaultCompiler()
          ->CompileAndLoad(std::move(hlo_program),
                           std::make_unique<xla::ifrt::XlaCompileOptions>(
                               xla_compile_options, device_list))
          .Await();
    }
    cache_miss_counter_++;
    cache_.insert(fingerprint);
    return value_fn(std::move(hlo_program),
                    std::make_unique<xla::ifrt::XlaCompileOptions>(
                        xla_compile_options, device_list));
  };

  uint64_t cache_hit_counter() { return cache_hit_counter_; };
  uint64_t cache_miss_counter() { return cache_miss_counter_; };

 private:
  absl::Mutex mu_;
  uint64_t cache_hit_counter_ = 0;
  uint64_t cache_miss_counter_ = 0;
  absl::flat_hash_set<uint64_t> cache_ ABSL_GUARDED_BY(mu_);
  std::shared_ptr<xla::ifrt::Client> ifrt_client_;
};

TEST(BuildIfrtExecutableTest, Ok) {
  const std::string kMlirPath = tsl::GetDataDependencyFilepath(
      "runtime/libs/ifrt/testdata/"
      "simple.mlir");

  StableHloLoader loader;
  auto module = loader.LoadFromFile(kMlirPath).value();
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(auto client, xla::ifrt::test_util::GetClient());

  BuildIfrtExecutableOptions options;
  options.num_replicas = 1;
  options.num_cores_per_replica = 1;
  options.device_assignment = *client->GetDefaultDeviceAssignment(1, 1);
  TF_ASSERT_OK(BuildIfrtExecutable(module.get(), options, client, nullptr));
}

TEST(BuildIfrtExecutableTest, InvalidOptionsShallFail) {
  const std::string kMlirPath = tsl::GetDataDependencyFilepath(
      "runtime/libs/ifrt/testdata/"
      "simple.mlir");

  StableHloLoader loader;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      loader.LoadFromFile(kMlirPath).value();
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(auto client, xla::ifrt::test_util::GetClient());

  BuildIfrtExecutableOptions options;
  options.num_replicas = 0;
  options.num_cores_per_replica = 0;
  options.device_assignment = xla::DeviceAssignment();
  EXPECT_THAT(BuildIfrtExecutable(module.get(), options, client, nullptr),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(BuildIfrtExecutableTest, PersistentCompilationCache) {
  const std::string kMlirPath = tsl::GetDataDependencyFilepath(
      "runtime/libs/ifrt/testdata/"
      "simple.mlir");

  StableHloLoader loader;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      loader.LoadFromFile(kMlirPath).value();
  ASSERT_TRUE(module);

  std::shared_ptr<xla::ifrt::MockClient> mock_client =
      std::make_shared<xla::ifrt::MockClient>();
  std::shared_ptr<xla::ifrt::MockCompiler> mock_compiler =
      std::make_shared<xla::ifrt::MockCompiler>();
  std::shared_ptr<xla::ifrt::MockDevice> mock_device =
      std::make_shared<xla::ifrt::MockDevice>();
  tsl::RCReference<xla::ifrt::MockDeviceList> mock_device_list =
      tsl::MakeRef<xla::ifrt::MockDeviceList>();
  xla::ifrt::DeviceListRef mock_device_list_ref =
      xla::ifrt::RCReferenceWrapper<xla::ifrt::DeviceList>(mock_device_list);
  std::vector<xla::ifrt::Device*> devices = {mock_device.get()};
  ON_CALL(*mock_device_list, devices())
      .WillByDefault(Return(absl::MakeConstSpan(devices)));
  ON_CALL(*mock_device_list, AddressableDeviceList())
      .WillByDefault(Return(mock_device_list.get()));

  EXPECT_CALL(*mock_client, GetDefaultCompiler())
      .WillRepeatedly(Return(mock_compiler.get()));

  xla::DeviceAssignment default_da(1, 1);
  default_da(0, 0) = 0;
  EXPECT_CALL(*mock_client, GetDefaultDeviceAssignment(1, 1))
      .WillRepeatedly(Return(default_da));
  EXPECT_CALL(*mock_client, LookupDevice(xla::ifrt::DeviceId(0)))
      .WillRepeatedly(Return(mock_device.get()));
  EXPECT_CALL(*mock_client,
              MakeDeviceList(testing::ElementsAre(mock_device.get())))
      .WillRepeatedly(Return(mock_device_list_ref));
  EXPECT_CALL(*mock_client, platform_id())
      .WillRepeatedly(Return(xla::PjRtPlatformId(xla::TpuId())));

  EXPECT_CALL(*mock_compiler, CompileAndLoad(_, _))
      .WillRepeatedly(
          [&](std::unique_ptr<xla::ifrt::Program> program,
              std::unique_ptr<xla::ifrt::CompileOptions> ifrt_options) {
            return xla::ifrt::LoadedExecutableRef(
                std::make_unique<xla::ifrt::MockLoadedExecutable>());
          });

  auto fake_cache =
      std::make_unique<FakePersistentCompilationCache>(mock_client);

  BuildIfrtExecutableOptions options;
  options.num_replicas = 1;
  options.num_cores_per_replica = 1;
  options.device_assignment = *mock_client->GetDefaultDeviceAssignment(1, 1);

  // First call to BuildIfrtExecutable should be a cache miss.
  TF_ASSERT_OK(BuildIfrtExecutable(module.get(), options, mock_client,
                                   fake_cache.get()));
  EXPECT_EQ(fake_cache->cache_miss_counter(), 1);
  EXPECT_EQ(fake_cache->cache_hit_counter(), 0);

  mlir::OwningOpRef<mlir::ModuleOp> module_dup =
      loader.LoadFromFile(kMlirPath).value();
  ASSERT_TRUE(module_dup);

  // Second call to BuildIfrtExecutable should be a cache hit.
  TF_ASSERT_OK(BuildIfrtExecutable(module_dup.get(), options, mock_client,
                                   fake_cache.get()));
  EXPECT_EQ(fake_cache->cache_miss_counter(), 1);
  EXPECT_EQ(fake_cache->cache_hit_counter(), 1);
}

TEST(BuildIfrtExecutableTest, CpuClientBypassesPersistentCompilationCache) {
  const std::string kMlirPath = tsl::GetDataDependencyFilepath(
      "runtime/libs/ifrt/testdata/"
      "simple.mlir");

  StableHloLoader loader;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      loader.LoadFromFile(kMlirPath).value();
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(auto client, xla::ifrt::test_util::GetClient());

  auto fake_cache = std::make_unique<FakePersistentCompilationCache>(client);

  BuildIfrtExecutableOptions options;
  options.num_replicas = 1;
  options.num_cores_per_replica = 1;
  options.device_assignment = *client->GetDefaultDeviceAssignment(1, 1);

  TF_ASSERT_OK(
      BuildIfrtExecutable(module.get(), options, client, fake_cache.get()));
  // Because client is a CPU client, FakePersistentCompilationCache should be
  // bypassed entirely.
  EXPECT_EQ(fake_cache->cache_miss_counter(), 0);
  EXPECT_EQ(fake_cache->cache_hit_counter(), 0);
}

TEST(BuildIfrtExecutableTest, CompEnvsNotStripped) {
#ifdef XLA_SERVING_ON_GCE
  GTEST_SKIP() << "CompEnvs are stripped when XLA_SERVING_ON_GCE is defined.";
#endif
  // Register a process new env fn for google::protobuf::StringValue so that
  // the CompilationEnvironments proto can be deserialized.
  // This is a hack for testing purposes.
  xla::CompilationEnvironments::RegisterProcessNewEnvFn(
      google::protobuf::StringValue::descriptor(),
      [](std::unique_ptr<tsl::protobuf::Message> env)
          -> absl::StatusOr<std::unique_ptr<tsl::protobuf::Message>> {
        return env;
      });
  const std::string kMlirPath = tsl::GetDataDependencyFilepath(
      "runtime/libs/ifrt/testdata/"
      "simple.mlir");

  StableHloLoader loader;
  mlir::OwningOpRef<mlir::ModuleOp> module =
      loader.LoadFromFile(kMlirPath).value();
  ASSERT_TRUE(module);

  std::shared_ptr<xla::ifrt::MockClient> mock_client =
      std::make_shared<xla::ifrt::MockClient>();
  std::shared_ptr<xla::ifrt::MockCompiler> mock_compiler =
      std::make_shared<xla::ifrt::MockCompiler>();
  std::shared_ptr<xla::ifrt::MockDevice> mock_device =
      std::make_shared<xla::ifrt::MockDevice>();
  tsl::RCReference<xla::ifrt::MockDeviceList> mock_device_list =
      tsl::MakeRef<xla::ifrt::MockDeviceList>();
  xla::ifrt::DeviceListRef mock_device_list_ref =
      xla::ifrt::RCReferenceWrapper<xla::ifrt::DeviceList>(mock_device_list);
  std::vector<xla::ifrt::Device*> devices = {mock_device.get()};
  ON_CALL(*mock_device_list, devices())
      .WillByDefault(Return(absl::MakeConstSpan(devices)));
  ON_CALL(*mock_device_list, AddressableDeviceList())
      .WillByDefault(Return(mock_device_list.get()));

  EXPECT_CALL(*mock_client, GetDefaultCompiler())
      .WillOnce(Return(mock_compiler.get()));

  xla::DeviceAssignment default_da(1, 1);
  default_da(0, 0) = 0;
  EXPECT_CALL(*mock_client, GetDefaultDeviceAssignment(1, 1))
      .WillOnce(Return(default_da));
  EXPECT_CALL(*mock_client, LookupDevice(xla::ifrt::DeviceId(0)))
      .WillOnce(Return(mock_device.get()));
  EXPECT_CALL(*mock_client,
              MakeDeviceList(testing::ElementsAre(mock_device.get())))
      .WillOnce(Return(mock_device_list_ref));

  BuildIfrtExecutableOptions options;
  options.num_replicas = 1;
  options.num_cores_per_replica = 1;
  options.device_assignment = *mock_client->GetDefaultDeviceAssignment(1, 1);
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
  EXPECT_CALL(*mock_compiler, CompileAndLoad(_, _))
      .WillOnce([&](std::unique_ptr<xla::ifrt::Program> program,
                    std::unique_ptr<xla::ifrt::CompileOptions> ifrt_options) {
        captured_ifrt_options = std::move(ifrt_options);
        return xla::ifrt::LoadedExecutableRef(
            std::make_unique<xla::ifrt::MockLoadedExecutable>());
      });

  ASSERT_OK(BuildIfrtExecutable(module.get(), options, mock_client,
                                /*persistent_compilation_cache=*/nullptr));
  ASSERT_NE(captured_ifrt_options, nullptr);
  auto* xla_ifrt_options = xla::ifrt::dyn_cast<xla::ifrt::XlaCompileOptions>(
      captured_ifrt_options.get());
  ASSERT_NE(xla_ifrt_options, nullptr);

  EXPECT_THAT(xla_ifrt_options->compile_options.executable_build_options
                  .mutable_comp_envs()
                  ->GetEnv<google::protobuf::StringValue>(),
              tsl::proto_testing::EqualsProto(test_env_value));
}

TEST(GetProgramFingerprintWithTimestampTest, Ok) {
  const std::string kMlirPath = tsl::GetDataDependencyFilepath(
      "runtime/libs/ifrt/testdata/"
      "simple.mlir");

  StableHloLoader loader;
  auto module = loader.LoadFromFile(kMlirPath).value();
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(auto client, xla::ifrt::test_util::GetClient());
  if (xla::ifrt::dyn_cast<xla::cpu::NanoIfrtClient>(client.get())) {
    GTEST_SKIP() << "Skipping test for NanoIfrtClient due to lack of "
                    "Fingerprint() support.";
  }

  BuildIfrtExecutableOptions options;
  options.num_replicas = 1;
  options.num_cores_per_replica = 1;
  options.device_assignment = *client->GetDefaultDeviceAssignment(1, 1);
  TF_ASSERT_OK_AND_ASSIGN(
      xla::ifrt::LoadedExecutableRef executable,
      BuildIfrtExecutable(module.get(), options, client, nullptr));

  const absl::Time now = absl::Now();
  std::string fp_with_ts = GetProgramFingerprintWithTimestamp(executable, now);
  std::vector<std::string> parts =
      absl::StrSplit(fp_with_ts, kProgramFingerprintAndTimestampDelimiter);
  ASSERT_EQ(parts.size(), 2);
  TF_ASSERT_OK_AND_ASSIGN(const auto fingerprint, executable->Fingerprint());
  ASSERT_TRUE(fingerprint.has_value());
  EXPECT_EQ(parts[0], *fingerprint);
  EXPECT_EQ(parts[1], absl::StrCat(absl::ToUnixMicros(now)));
}

TEST(GetProgramFingerprintWithTimestampTest, FingerprintNotOk) {
  auto executable = std::make_shared<xla::ifrt::MockLoadedExecutable>();
  EXPECT_CALL(*executable, Fingerprint())
      .WillOnce(Return(absl::InternalError("Fingerprint() failed")));

  const absl::Time now = absl::Now();
  std::string fp_with_ts = GetProgramFingerprintWithTimestamp(executable, now);
  std::vector<std::string> parts =
      absl::StrSplit(fp_with_ts, kProgramFingerprintAndTimestampDelimiter);
  ASSERT_EQ(parts.size(), 2);
  EXPECT_EQ(parts[0], kMissingProgramFingerprint);
  EXPECT_EQ(parts[1], absl::StrCat(absl::ToUnixMicros(now)));
}

TEST(GetProgramFingerprintWithTimestampTest, FingerprintHasNoValue) {
  auto executable = std::make_shared<xla::ifrt::MockLoadedExecutable>();
  EXPECT_CALL(*executable, Fingerprint()).WillOnce(Return(std::nullopt));

  const absl::Time now = absl::Now();
  std::string fp_with_ts = GetProgramFingerprintWithTimestamp(executable, now);
  std::vector<std::string> parts =
      absl::StrSplit(fp_with_ts, kProgramFingerprintAndTimestampDelimiter);
  ASSERT_EQ(parts.size(), 2);
  EXPECT_EQ(parts[0], kMissingProgramFingerprint);
  EXPECT_EQ(parts[1], absl::StrCat(absl::ToUnixMicros(now)));
}

}  // namespace
}  // namespace xla_serving
