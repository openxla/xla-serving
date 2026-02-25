#include "third_party/neptune/runtime/libs/ifrt/executable_build_utils.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "devtools/build/runtime/get_runfiles_dir.h"
#include "google/protobuf/any.proto.h"
#include "google/protobuf/wrappers.proto.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/base/thread_annotations.h"
#include "third_party/absl/container/flat_hash_set.h"
#include "third_party/absl/functional/any_invocable.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/status_matchers.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/str_split.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/synchronization/mutex.h"
#include "third_party/absl/time/clock.h"
#include "third_party/absl/time/time.h"
#include "third_party/absl/types/span.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/Support/Casting.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/BuiltinOps.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/OwningOpRef.h"
#include "third_party/neptune/runtime/libs/ifrt/persistent_compilation_cache/ifrt_persistent_compilation_cache.h"
#include "third_party/neptune/runtime/libs/ifrt/stablehlo_loader.h"
#include "third_party/tensorflow/compiler/xla/backends/cpu/nanort/ifrt_client.h"
#include "third_party/tensorflow/compiler/xla/pjrt/pjrt_executable.h"
#include "third_party/tensorflow/compiler/xla/pjrt/proto/compile_options.proto.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/client.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/compiler.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/device.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/device_list.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/executable.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/hlo/hlo_program.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/mock.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/program.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/ref_wrapper.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/test_util.h"
#include "third_party/tensorflow/compiler/xla/python/pjrt_ifrt/xla_compiler.h"
#include "third_party/tensorflow/compiler/xla/service/compilation_environments.h"
#include "third_party/tensorflow/compiler/xla/service/computation_placer.h"
#include "third_party/tensorflow/compiler/xla/tsl/concurrency/ref_count.h"
#include "third_party/tensorflow/compiler/xla/tsl/lib/core/status_test_util.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"

namespace jsv {
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
    uint64_t fingerprint = hlo_program->Fingerprint();
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
  const std::string kMlirPath = devtools_build::GetDataDependencyFilepath(
      "google3/third_party/neptune/runtime/libs/ifrt/testdata/"
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
  const std::string kMlirPath = devtools_build::GetDataDependencyFilepath(
      "google3/third_party/neptune/runtime/libs/ifrt/testdata/"
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
  const std::string kMlirPath = devtools_build::GetDataDependencyFilepath(
      "google3/third_party/neptune/runtime/libs/ifrt/testdata/"
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

  // First call to BuildIfrtExecutable should be a cache miss.
  TF_ASSERT_OK(
      BuildIfrtExecutable(module.get(), options, client, fake_cache.get()));
  EXPECT_EQ(fake_cache->cache_miss_counter(), 1);
  EXPECT_EQ(fake_cache->cache_hit_counter(), 0);

  mlir::OwningOpRef<mlir::ModuleOp> module_dup =
      loader.LoadFromFile(kMlirPath).value();
  ASSERT_TRUE(module_dup);

  // Second call to BuildIfrtExecutable should be a cache hit.
  TF_ASSERT_OK(
      BuildIfrtExecutable(module_dup.get(), options, client, fake_cache.get()));
  EXPECT_EQ(fake_cache->cache_miss_counter(), 1);
  EXPECT_EQ(fake_cache->cache_hit_counter(), 1);
}

TEST(BuildIfrtExecutableTest, CompEnvsNotStripped) {
  // Register a process new env fn for google::protobuf::StringValue so that
  // the CompilationEnvironments proto can be deserialized.
  // This is a hack for testing purposes.
  xla::CompilationEnvironments::RegisterProcessNewEnvFn(
      google::protobuf::StringValue::descriptor(),
      [](std::unique_ptr<tsl::protobuf::Message> env)
          -> absl::StatusOr<std::unique_ptr<tsl::protobuf::Message>> {
        return env;
      });
  const std::string kMlirPath = devtools_build::GetDataDependencyFilepath(
      "google3/third_party/neptune/runtime/libs/ifrt/testdata/"
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
  auto* xla_ifrt_options =
      llvm::dyn_cast<xla::ifrt::XlaCompileOptions>(captured_ifrt_options.get());
  ASSERT_NE(xla_ifrt_options, nullptr);

  EXPECT_THAT(xla_ifrt_options->compile_options.executable_build_options
                  .mutable_comp_envs()
                  ->GetEnv<google::protobuf::StringValue>(),
              testing::EqualsProto(test_env_value));
}

TEST(GetProgramFingerprintWithTimestampTest, Ok) {
  const std::string kMlirPath = devtools_build::GetDataDependencyFilepath(
      "google3/third_party/neptune/runtime/libs/ifrt/testdata/"
      "simple.mlir");

  StableHloLoader loader;
  auto module = loader.LoadFromFile(kMlirPath).value();
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(auto client, xla::ifrt::test_util::GetClient());
  if (llvm::dyn_cast<xla::cpu::NanoIfrtClient>(client.get())) {
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
  ASSERT_OK_AND_ASSIGN(const auto fingerprint, executable->Fingerprint());
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
}  // namespace jsv
