#include "third_party/neptune/runtime/serving/gcp/server/orbax_grpc_server.h"

#include <memory>
#include <string>
#include <utility>

#include "devtools/build/runtime/get_runfiles_dir.h"
#include "net/util/ports.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/flags/declare.h"
#include "third_party/absl/flags/flag.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/grpc/include/grpcpp/channel.h"
#include "third_party/grpc/include/grpcpp/security/server_credentials.h"
#include "third_party/grpc/include/grpcpp/server.h"
#include "third_party/grpc/include/grpcpp/server_builder.h"
#include "third_party/grpc/include/grpcpp/server_context.h"
#include "third_party/grpc/include/grpcpp/support/channel_arguments.h"
#include "third_party/grpc/include/grpcpp/support/status.h"
#include "third_party/neptune/runtime/core/host_buffer.h"
#include "third_party/neptune/runtime/core/host_buffer_matcher.h"
#include "third_party/neptune/runtime/core/proto/host_buffer.proto.h"
#include "third_party/neptune/runtime/serving/gcp/proto/model.proto.h"
#include "third_party/neptune/runtime/serving/gcp/proto/predict.proto.h"
#include "third_party/neptune/runtime/serving/gcp/proto/prediction_service.grpc.pb.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/dtype.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/shape.h"

ABSL_DECLARE_FLAG(std::string, orbax_grpc_server_platform);

namespace jsv {
namespace serving {
namespace {

using ::testing::Eq;
using ::testing::NotNull;
using ::testing::SizeIs;
using ::testing::status::IsOk;

// Test fixture for OrbaxGrpcServer.
class OrbaxGrpcServerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::string platform = absl::GetFlag(FLAGS_orbax_grpc_server_platform);
    if (platform == "gpu") {
      model_path_ = devtools_build::GetDataDependencyFilepath(
          "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
          "testdata/linear_model_2x1_sharded_gen");
      model_name_ = "linear_model_2x1_sharded";
    } else {
      model_path_ = devtools_build::GetDataDependencyFilepath(
          "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
          "testdata/linear_model_gen");
      model_name_ = "linear_model";
    }

    // Set up the gRPC server.
    int port = net_util::PickUnusedPortOrDie();
    server_address_ = absl::StrCat("localhost:", port);

    ASSERT_OK_AND_ASSIGN(server_,
                         OrbaxGrpcServer::Create(model_name_, model_path_));
    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address_,
                             grpc::InsecureServerCredentials());
    builder.RegisterService(server_.get());
    grpc_server_ = builder.BuildAndStart();
    ASSERT_THAT(grpc_server_, NotNull());

    // Set up the gRPC client stub.
    std::shared_ptr<grpc::Channel> channel =
        grpc_server_->InProcessChannel(grpc::ChannelArguments());
    stub_ = PredictionService::NewStub(channel);
    ASSERT_THAT(stub_, NotNull());
  }

  void TearDown() override {
    if (grpc_server_) {
      grpc_server_->Shutdown();
      grpc_server_->Wait();
    }
  }

  std::string model_path_;
  std::string model_name_;
  std::string server_address_;
  std::unique_ptr<OrbaxGrpcServer> server_;
  std::unique_ptr<grpc::Server> grpc_server_;
  std::unique_ptr<PredictionService::Stub> stub_;
};

TEST_F(OrbaxGrpcServerTest, PredictLinearModelSucceeds) {
  // Prepare the PredictRequest. The linear model expects an input "x".
  PredictRequest request;
  request.mutable_model_spec()->set_name(model_name_);
  request.mutable_model_spec()->set_signature_name("serving_default");

  jsv::HostBuffer input_buffer(xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
                               xla::ifrt::Shape({4, 2}),
                               /*allocator=*/nullptr);
  auto input_data = input_buffer.flat<float>();
  for (int i = 0; i < 8; ++i) {
    input_data(i) = static_cast<float>(i);
  }

  (*request.mutable_inputs())["x"] =
      jsv::HostBuffer::ToHostBufferProto(std::move(input_buffer));

  grpc::ClientContext context;
  PredictResponse response;
  grpc::Status status = stub_->Predict(&context, request, &response);
  ASSERT_THAT(status, IsOk()) << status.error_message();

  // Verify the response.
  EXPECT_THAT(response.model_spec().name(), Eq(model_name_));
  EXPECT_THAT(response.model_spec().signature_name(), Eq("serving_default"));
  ASSERT_THAT(response.outputs(), SizeIs(1));
  ASSERT_TRUE(response.outputs().contains("y"));

  const HostBufferProto& output_proto = response.outputs().at("y");
  ASSERT_OK_AND_ASSIGN(jsv::HostBuffer output_host_buffer,
                       jsv::HostBuffer::FromProto(output_proto));

  jsv::HostBuffer expected_host_buffer(
      xla::ifrt::DType(xla::ifrt::DType::Kind::kF32), xla::ifrt::Shape({4, 2}));
  auto expected_data = expected_host_buffer.flat<float>();
  float expected_values[] = {2, 3, 6, 7, 10, 11, 14, 15};
  for (int i = 0; i < 8; ++i) {
    expected_data(i) = expected_values[i];
  }

  EXPECT_THAT(output_host_buffer, HostBufferEq(expected_host_buffer));
}

}  // namespace
}  // namespace serving
}  // namespace jsv
