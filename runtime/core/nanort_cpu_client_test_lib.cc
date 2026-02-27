#include <memory>

#include "third_party/absl/status/statusor.h"
#include "third_party/tensorflow/compiler/xla/backends/cpu/nanort/ifrt_client.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/client.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/test_util.h"

namespace jsv {
namespace {

const bool kUnused =
    (xla::ifrt::test_util::RegisterClientFactory(
         []() -> absl::StatusOr<std::shared_ptr<xla::ifrt::Client>> {
           return xla::cpu::NanoIfrtClient::CreateWithDevices(8);
         }),
     true);

}  // namespace
}  // namespace jsv
