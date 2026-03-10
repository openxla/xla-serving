#include <memory>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "xla/backends/cpu/nanort/ifrt_client.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/test_util.h"  // from @xla

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
