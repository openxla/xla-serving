#include "runtime/interpreter/interpreter_test_base.h"  // from @neptune

#include <memory>
#include <string>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "xla/python/ifrt/test_util.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla

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
