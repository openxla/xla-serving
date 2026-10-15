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
#include <utility>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"  // from @xla
#include "xla/pjrt/pjrt_client.h"  // from @xla
#include "xla/pjrt/plugin/plugin_names.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/test_util.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_client.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla

namespace xla_serving {
namespace {

const bool kUnused =
    (xla::ifrt::test_util::RegisterClientFactory(
         []() -> absl::StatusOr<std::shared_ptr<xla::ifrt::Client>> {
           TF_ASSIGN_OR_RETURN(
               std::unique_ptr<xla::PjRtClient> pjrt_client,
               xla::GetCApiClient(kCpuPjrtName,
                                  /*create_options=*/{}));
           xla::ifrt::PjRtClient::CreateOptions options;
           options.pjrt_client = std::move(pjrt_client);
           TF_ASSIGN_OR_RETURN(
               std::unique_ptr<xla::ifrt::PjRtClient> ifrt_client,
               xla::ifrt::PjRtClient::Create(std::move(options)));
           return std::shared_ptr<xla::ifrt::Client>(ifrt_client.release());
         }),
     true);

}  // namespace
}  // namespace xla_serving
