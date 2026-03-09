#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_INTERPRETER_TEST_BASE_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_INTERPRETER_TEST_BASE_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/flags/declare.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/neptune/runtime/core/value.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/client.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/threadpool.h"
#include "third_party/tensorflow/compiler/xla/xla_data.proto.h"

ABSL_DECLARE_FLAG(std::string, jsv_testonly_platform_name_override);

namespace jsv {

class InterpreterTestBase : public testing::Test {
 protected:
  void SetUp() override;

 protected:
  static constexpr int kThreadPoolNumThreads = 16;

  std::shared_ptr<xla::ifrt::Client> client_;
  std::unique_ptr<tsl::thread::ThreadPool> execute_worker_thread_pool_;
  std::unique_ptr<tsl::thread::ThreadPool> computation_thread_pool_;
};

struct InterpreterMatchOutputTestParam {
  absl::string_view testcase_name;
  absl::string_view model_name;
  absl::string_view model_path;
  absl::string_view pipeline_name;
  std::vector<Value> inputs;
  std::vector<testing::Matcher<const Value&>> expected_outputs;
  bool is_gpu_test_supported = false;
};

struct InterpreterMapIoTestParam {
  absl::string_view testcase_name;
  absl::string_view model_name;
  absl::string_view model_path;
  absl::string_view pipeline_name;
  absl::flat_hash_map<std::string, Value> inputs;
  std::vector<testing::Matcher<std::pair<const std::string, const Value&>>>
      expected_outputs;
  bool is_gpu_test_supported = false;
};

}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_INTERPRETER_TEST_BASE_H_
