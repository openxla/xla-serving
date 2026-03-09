#include "third_party/neptune/runtime/interpreter/interpreter_test_base.h"

#include <memory>
#include <string>

#include "third_party/absl/flags/flag.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/test_util.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/env.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/threadpool.h"
#include "third_party/tensorflow/compiler/xla/xla_data.proto.h"

ABSL_FLAG(std::string, jsv_testonly_platform_name_override, "",
          "When non-empty, overrides the platform name used in tests."
          " The value should match one of the enum names in "
          " orbax_model_manifest::Platform.");

namespace jsv {

void InterpreterTestBase::SetUp() {
  TF_ASSERT_OK_AND_ASSIGN(client_, xla::ifrt::test_util::GetClient());

  execute_worker_thread_pool_ = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "OrbaxExecutionThreadPool",
      kThreadPoolNumThreads);
  computation_thread_pool_ = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "OrbaxComputationThreadPool",
      kThreadPoolNumThreads);
}

}  // namespace jsv
