// TODO: Move this test file to
// third_party/neptune/data_loader/ocp_loader/checkpoint_loader_test.cc.
#include "third_party/neptune/data_loader/ocp_loader/checkpoint_loader.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "devtools/build/runtime/get_runfiles_dir.h"
#include "learning/infra/mira/experimental/orbax_model/core/host_buffer.h"
#include "learning/infra/mira/experimental/orbax_model/core/host_buffer_matcher.h"
#include "learning/infra/mira/experimental/orbax_model/core/host_buffer_test_utils.h"
#include "learning/infra/mira/experimental/orbax_model/core/sharding_utils.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/log/log.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/types/span.h"
#include "third_party/neptune/data_loader/ocp_loader/common_types.h"
#include "third_party/py/orbax/experimental/model/core/protos/manifest.proto.h"
#include "third_party/tensorflow/compiler/xla/hlo/ir/hlo_sharding.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/array.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/client.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/dtype.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/shape.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/test_util.h"
#include "third_party/tensorflow/compiler/xla/tsl/concurrency/future.h"
#include "third_party/tensorflow/compiler/xla/tsl/concurrency/ref_count.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/env.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/test.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/threadpool.h"
#include "third_party/tensorflow/compiler/xla/xla_data.proto.h"

namespace jsv {
namespace {

using ::testing::HasSubstr;
using ::testing::status::StatusIs;

struct CheckpointLoaderTestParam {
  neptune::CheckpointType checkpoint_type;
  std::string root_path;
  std::vector<neptune::ArraySpec> weight_specs;
  std::vector<std::string> weight_names;
  std::vector<int> device_ids;
  std::vector<HostBuffer> expected_out_host_buffer;
};

using CheckpointLoaderTest =
    ::testing::TestWithParam<CheckpointLoaderTestParam>;

// Wrapper functions for building sharding specs for a given shape with a
// natural device order.
xla::OpSharding Tile(absl::Span<const int64_t> dims) {
  return xla::HloSharding::IotaTile(dims).ToProto();
}
xla::OpSharding Replicate() { return xla::HloSharding::Replicate().ToProto(); }

TEST_P(CheckpointLoaderTest, basic) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "Resharding", /*num_threads=*/4);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());
  TF_ASSERT_OK_AND_ASSIGN(
      auto device_list,
      xla::ifrt::test_util::GetDevices(client.get(), GetParam().device_ids));
  LOG(INFO) << " size of devices " << device_list->devices().size();

  TF_ASSERT_OK_AND_ASSIGN(std::vector<xla::ifrt::ArrayRef> results,
                          neptune::LoadCheckpoint(
                              GetParam().root_path, GetParam().weight_names,
                              GetParam().weight_specs, device_list, *client,
                              GetParam().checkpoint_type));
  for (int i = 0; i < results.size(); ++i) {
    SCOPED_TRACE(absl::StrCat("Array ", i, " of ", results.size()));
    TF_ASSERT_OK_AND_ASSIGN(
        auto hlo_sharding,
        xla::HloSharding::FromProto(GetParam().weight_specs[i].sharding));
    tsl::Future<HostBuffer> host_buffer_future =
        sharding_internal::MakeHostBufferFromArray(
            *client, *results[i], hlo_sharding, device_list, thread_pool);
    TF_ASSERT_OK_AND_ASSIGN(auto host_buffer, host_buffer_future.Await());

    EXPECT_THAT(host_buffer,
                HostBufferEq(GetParam().expected_out_host_buffer[i]));
  }
}

INSTANTIATE_TEST_SUITE_P(
    CheckpointLoaderTests, CheckpointLoaderTest,
    ::testing::ValuesIn<CheckpointLoaderTestParam>({
        {
            .checkpoint_type = neptune::CheckpointType::kStandard,
            .root_path = devtools_build::GetDataDependencyFilepath(
                "google3/third_party/neptune/"
                "data_loader/testdata/non_spmd_checkpoint_gen"),
            .weight_specs =
                {
                    neptune::ArraySpec{
                        .dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                        .shape = xla::ifrt::Shape({4}),
                        .sharding = Replicate(),
                    },
                    neptune::ArraySpec{
                        .dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                        .shape = xla::ifrt::Shape({8}),
                        .sharding = Replicate(),
                    },
                },
            .weight_names = {"params/0", "params/1"},
            .device_ids = {0},
            .expected_out_host_buffer =
                {
                    AsHostBuffer<float>({0, 1, 2, 3}, xla::ifrt::Shape({4})),
                    AsHostBuffer<float>({4, 5, 6, 7, 8, 9, 10, 11},
                                        xla::ifrt::Shape({8})),
                },
        },
        {
            .checkpoint_type = neptune::CheckpointType::kStandard,
            .root_path = devtools_build::GetDataDependencyFilepath(
                "google3/third_party/neptune/"
                "data_loader/testdata/spmd_checkpoint_gen"),
            .weight_specs =
                {
                    neptune::ArraySpec{
                        .dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                        .shape = xla::ifrt::Shape({2, 2}),
                        .sharding = Tile({2, 2}),
                    },
                    neptune::ArraySpec{
                        .dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                        .shape = xla::ifrt::Shape({2, 4}),
                        .sharding = Tile({2, 2}),
                    },
                },
            .weight_names = {"params/0", "params/1"},
            .device_ids = {0, 1, 2, 3},
            .expected_out_host_buffer =
                {
                    AsHostBuffer<float>({0, 1, 2, 3}, xla::ifrt::Shape({2, 2})),
                    AsHostBuffer<float>({4, 5, 6, 7, 8, 9, 10, 11},
                                        xla::ifrt::Shape({2, 4})),
                },
        },
        {
            .checkpoint_type = neptune::CheckpointType::kOcdbt,
            .root_path = devtools_build::GetDataDependencyFilepath(
                "google3/third_party/neptune/"
                "data_loader/testdata/non_spmd_ocdbt_gen"),
            .weight_specs =
                {
                    neptune::ArraySpec{
                        .dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                        .shape = xla::ifrt::Shape({4}),
                        .sharding = Replicate(),
                    },
                    neptune::ArraySpec{
                        .dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                        .shape = xla::ifrt::Shape({8}),
                        .sharding = Replicate(),
                    },
                },
            .weight_names = {"param_0", "param_1"},
            .device_ids = {0},
            .expected_out_host_buffer =
                {
                    AsHostBuffer<float>({0, 1, 2, 3}, xla::ifrt::Shape({4})),
                    AsHostBuffer<float>({4, 5, 6, 7, 8, 9, 10, 11},
                                        xla::ifrt::Shape({8})),
                },
        },
        {
            .checkpoint_type = neptune::CheckpointType::kOcdbt,
            .root_path = devtools_build::GetDataDependencyFilepath(
                "google3/third_party/neptune/"
                "data_loader/testdata/spmd_ocdbt_gen"),
            .weight_specs =
                {
                    neptune::ArraySpec{
                        .dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                        .shape = xla::ifrt::Shape({2, 2}),
                        .sharding = Tile({2, 2}),
                    },
                    neptune::ArraySpec{
                        .dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                        .shape = xla::ifrt::Shape({2, 4}),
                        .sharding = Tile({2, 2}),
                    },
                },
            .weight_names = {"param_0", "param_1"},
            .device_ids = {0, 1, 2, 3},
            .expected_out_host_buffer =
                {
                    AsHostBuffer<float>({0, 1, 2, 3}, xla::ifrt::Shape({2, 2})),
                    AsHostBuffer<float>({4, 5, 6, 7, 8, 9, 10, 11},
                                        xla::ifrt::Shape({2, 4})),
                },
        },
    }));

TEST(CheckpointLoaderFailTest, DuplicateWeightNames) {
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());
  auto results = neptune::LoadCheckpoint(
      /*root_path=*/"", /*weight_names=*/{"params/0", "params/0", "params/1"},
      /*weight_specs=*/{}, /*device_list=*/{}, *client,
      neptune::CheckpointType::kStandard);
  EXPECT_THAT(
      results.status(),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("Duplicate weight names found: params/0 (count: 2)")));
}

}  // namespace
}  // namespace jsv
