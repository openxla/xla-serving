#include "third_party/neptune/runtime/core/batching/threadsafe_status.h"

#include <utility>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/status/status.h"

namespace jsv {
namespace {

TEST(ThreadSafeStatus, DefaultOk) {
  ThreadSafeStatus status;
  EXPECT_OK(status.status());
}

TEST(ThreadSafeStatus, Update) {
  ThreadSafeStatus status;
  EXPECT_OK(status.status());

  status.Update(absl::FailedPreconditionError("original error"));
  EXPECT_EQ(status.status().code(), absl::StatusCode::kFailedPrecondition);

  status.Update(absl::OkStatus());
  EXPECT_EQ(status.status().code(), absl::StatusCode::kFailedPrecondition);

  status.Update(absl::InternalError("new error"));
  EXPECT_EQ(status.status().code(), absl::StatusCode::kFailedPrecondition);
}

TEST(ThreadSafeStatus, Move) {
  ThreadSafeStatus status;
  EXPECT_OK(std::move(status).status());
}

}  // namespace
}  // namespace jsv
