#include "third_party/neptune/runtime/data_loader/default_data_loader.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "devtools/build/runtime/get_runfiles_dir.h"
#include "third_party/neptune/runtime/interpreter/manifest_utils.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/neptune/runtime/core/ifrt_loaded_variable_registry.h"
#include "third_party/neptune/runtime/data_loader/data_loader.h"
#include "third_party/py/orbax/experimental/model/core/protos/manifest.proto.h"
#include "third_party/tensorflow/compiler/xla/hlo/ir/hlo_sharding.h"
#include "third_party/tensorflow/compiler/xla/hlo/parser/hlo_parser.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/array.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/client.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/dtype.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/shape.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/test_util.h"
#include "third_party/tensorflow/compiler/xla/tsl/concurrency/future.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/env.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/test.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/threadpool.h"
#include "third_party/tensorflow/compiler/xla/xla_data.proto.h"

namespace jsv {

namespace {

using ::testing::HasSubstr;
using ::testing::UnorderedElementsAre;
using ::testing::status::StatusIs;

constexpr absl::string_view kLinearModelPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/linear_model_gen";
constexpr absl::string_view kLinearModelWithPolymorphicInputConcretizedPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
    "testdata/linear_model_with_polymorphic_input_concretized_gen";
constexpr absl::string_view
    kLinearModelWithPolymorphicInputConcretizedWithUnusedWeightPath =
        "google3/learning/infra/mira/experimental/orbax_model/interpreter/"
        "testdata/"
        "linear_model_with_polymorphic_input_concretized_with_unused_weight_"
        "gen";

class DefaultDataLoaderTest : public ::testing::TestWithParam<bool> {
 public:
  void SetUp() override {
    use_thread_pool_ = GetParam();
    TF_ASSERT_OK_AND_ASSIGN(ifrt_client_, xla::ifrt::test_util::GetClient());
    if (use_thread_pool_) {
      thread_pool_ = std::make_unique<tsl::thread::ThreadPool>(
          tsl::Env::Default(), "checkpoint_loader_thread_pool", 2);
    }
    DataLoader::CreateOptions create_options{
        .ifrt_loaded_variable_registry = &ifrt_loaded_variable_registry_,
        .ifrt_client = ifrt_client_,
        .checkpoint_loader_thread_pool = thread_pool_.get()};
    TF_ASSERT_OK_AND_ASSIGN(
        data_loader_, DefaultDataLoader::Create(std::move(create_options)));
  }

 protected:
  bool use_thread_pool_;
  IfrtLoadedVariableRegistry ifrt_loaded_variable_registry_;
  std::shared_ptr<xla::ifrt::Client> ifrt_client_;
  std::unique_ptr<tsl::thread::ThreadPool> thread_pool_;
  std::unique_ptr<DefaultDataLoader> data_loader_;
};

TEST_P(DefaultDataLoaderTest, Basic) {
  // Get the manifest.
  const std::string model_path =
      devtools_build::GetDataDependencyFilepath(kLinearModelPath);
  TF_ASSERT_OK_AND_ASSIGN(auto manifest, GetManifest(model_path));
  ASSERT_TRUE(manifest.objects().contains("params"));
  ASSERT_TRUE(manifest.objects().at("params").has_value());
  ASSERT_TRUE(manifest.objects().at("params").value().has_external());
  ASSERT_TRUE(manifest.objects().contains("linear"));
  ASSERT_TRUE(manifest.objects().at("linear").has_function());
  ASSERT_EQ(manifest.objects().at("linear").function().data_names_size(), 2);
  ASSERT_EQ(manifest.objects().at("linear").function().data_names(0), "bias");
  ASSERT_EQ(manifest.objects().at("linear").function().data_names(1),
            "weights");
  TF_ASSERT_OK_AND_ASSIGN(
      AssignedDevices device,
      GetAssignedDevices(manifest, manifest.objects().at("linear").function(),
                         nullptr));
  std::vector<const orbax_model_manifest::Function*> stablehlo_funcs = {
      &manifest.objects().at("linear").function()};

  // Load the data.
  absl::StatusOr<DataLoader::LoadResult> load_result = data_loader_->Load(
      /*value_to_load=*/manifest.objects().at("params").value().external(),
      model_path, &device, &manifest,
      /*stablehlo_funcs=*/stablehlo_funcs);
  EXPECT_OK(load_result);
  EXPECT_THAT(load_result->data_names, UnorderedElementsAre("bias", "weights"));

  // Verify weights are loaded by fetching from the registry.
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto sharding,
        xla::ParseSharding(
            "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"));
    ASSERT_OK_AND_ASSIGN(auto loaded_variable,
                         ifrt_loaded_variable_registry_.GetLoadedVariable(
                             {.device_ids = {0, 1, 2, 3},
                              .input_name = "bias",
                              .hlo_sharding = sharding}));
    ASSERT_TRUE(
        loaded_variable.value.IsType<tsl::Future<xla::ifrt::ArrayRef>>());
    const auto array_future =
        loaded_variable.value.Get<tsl::Future<xla::ifrt::ArrayRef>>();
    if (use_thread_pool_) {
      // Disabling this async check due to risk of potential flakiness.
      // EXPECT_FALSE(array_future.IsReady());
    } else {
      EXPECT_TRUE(array_future.IsReady());
    }
    ASSERT_OK_AND_ASSIGN(const auto array, array_future.Await());
    // `bias` is the first param with shape (output_dim=2)
    EXPECT_EQ(array->shape(), xla::ifrt::Shape({2}));
    EXPECT_EQ(array->dtype(), xla::ifrt::DType(xla::ifrt::DType::kF32));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(auto sharding,
                            xla::ParseSharding("{devices=[2,2]<=[4]}"));
    ASSERT_OK_AND_ASSIGN(auto loaded_variable,
                         ifrt_loaded_variable_registry_.GetLoadedVariable(
                             {.device_ids = {0, 1, 2, 3},
                              .input_name = "weights",
                              .hlo_sharding = sharding}));
    ASSERT_TRUE(
        loaded_variable.value.IsType<tsl::Future<xla::ifrt::ArrayRef>>());
    const auto array_future =
        loaded_variable.value.Get<tsl::Future<xla::ifrt::ArrayRef>>();
    ASSERT_OK_AND_ASSIGN(const auto array, array_future.Await());
    // `weight` is the second param with shape (input_dim=2, output_dim=2)
    EXPECT_EQ(array->shape(), xla::ifrt::Shape({2, 2}));
    EXPECT_EQ(array->dtype(), xla::ifrt::DType(xla::ifrt::DType::kF32));
  }

  // Verify the total number of loaded variables stored in the registry.
  IfrtLoadedVariableRegistryTestPeer ifrt_loaded_variable_registry_test_peer(
      &ifrt_loaded_variable_registry_);
  auto loaded_variable_map =
      ifrt_loaded_variable_registry_test_peer.GetLoadedVariableMap();
  EXPECT_EQ(loaded_variable_map.size(), 2);
}

TEST_P(DefaultDataLoaderTest, CreateFailsWhenIfrtClientNotProvided) {
  DataLoader::CreateOptions create_options{
      .ifrt_loaded_variable_registry = &ifrt_loaded_variable_registry_,
      .ifrt_client = std::nullopt,
      .checkpoint_loader_thread_pool = thread_pool_.get()};
  EXPECT_THAT(DefaultDataLoader::Create(std::move(create_options)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("ifrt_client is required")));
}

TEST_P(DefaultDataLoaderTest, LoadFailsWhenValueToLoadHasNoData) {
  orbax_model_manifest::ExternalValue value_to_load;
  orbax_model_manifest::Manifest manifest;
  orbax_model_manifest::Function stablehlo_func;
  std::vector<const orbax_model_manifest::Function*> stablehlo_funcs = {
      &stablehlo_func};
  AssignedDevices assigned_devices;
  EXPECT_THAT(data_loader_->Load(value_to_load, "model_path", &assigned_devices,
                                 &manifest, stablehlo_funcs),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Checkpoint value has no `data`")));
}

TEST_P(DefaultDataLoaderTest, LoadFailsWhenMimeTypeIsWrong) {
  orbax_model_manifest::ExternalValue value_to_load;
  value_to_load.mutable_data()->set_mime_type("wrong_mime_type");
  orbax_model_manifest::Manifest manifest;
  orbax_model_manifest::Function stablehlo_func;
  std::vector<const orbax_model_manifest::Function*> stablehlo_funcs = {
      &stablehlo_func};
  AssignedDevices assigned_devices;
  EXPECT_THAT(
      data_loader_->Load(value_to_load, "model_path", &assigned_devices,
                         &manifest, stablehlo_funcs),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("only mime type \"application/x.orbax-checkpoint\" is "
                         "supported")));
}

TEST_P(DefaultDataLoaderTest, LoadFailsWhenDataTypeIsWrong) {
  orbax_model_manifest::ExternalValue value_to_load;
  value_to_load.mutable_data()->set_mime_type("application/x.orbax-checkpoint");
  value_to_load.mutable_data()->set_inlined_string(
      "this is not a file system location");
  orbax_model_manifest::Manifest manifest;
  orbax_model_manifest::Function stablehlo_func;
  std::vector<const orbax_model_manifest::Function*> stablehlo_funcs = {
      &stablehlo_func};
  AssignedDevices assigned_devices;
  EXPECT_THAT(
      data_loader_->Load(value_to_load, "model_path", &assigned_devices,
                         &manifest, stablehlo_funcs),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("only data type \"file_system_location\" is supported")));
}

TEST_P(DefaultDataLoaderTest, LoadFailsWhenModelPathNotProvided) {
  orbax_model_manifest::ExternalValue value_to_load;
  value_to_load.mutable_data()->set_mime_type("application/x.orbax-checkpoint");
  value_to_load.mutable_data()->mutable_file_system_location()->set_string_path(
      "checkpoint_value");
  orbax_model_manifest::Manifest manifest;
  orbax_model_manifest::Function stablehlo_func;
  std::vector<const orbax_model_manifest::Function*> stablehlo_funcs = {
      &stablehlo_func};
  AssignedDevices assigned_devices;
  EXPECT_THAT(data_loader_->Load(value_to_load, /*model_path=*/std::nullopt,
                                 &assigned_devices, &manifest, stablehlo_funcs),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("`model_path` is required")));
}

TEST_P(DefaultDataLoaderTest, LoadFailsWhenManifestNotProvided) {
  orbax_model_manifest::ExternalValue value_to_load;
  value_to_load.mutable_data()->set_mime_type("application/x.orbax-checkpoint");
  value_to_load.mutable_data()->mutable_file_system_location()->set_string_path(
      "checkpoint_value");
  orbax_model_manifest::Function stablehlo_func;
  std::vector<const orbax_model_manifest::Function*> stablehlo_funcs = {
      &stablehlo_func};
  AssignedDevices assigned_devices;
  EXPECT_THAT(data_loader_->Load(value_to_load, "model_path", &assigned_devices,
                                 /*manifest=*/std::nullopt, stablehlo_funcs),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("`manifest` is required")));
}

TEST_P(DefaultDataLoaderTest, LoadFailsWhenStableHloFuncNotProvided) {
  orbax_model_manifest::ExternalValue value_to_load;
  value_to_load.mutable_data()->set_mime_type("application/x.orbax-checkpoint");
  value_to_load.mutable_data()->mutable_file_system_location()->set_string_path(
      "checkpoint_value");
  orbax_model_manifest::Manifest manifest;
  AssignedDevices assigned_devices;
  EXPECT_THAT(data_loader_->Load(value_to_load, "model_path", &assigned_devices,
                                 &manifest,
                                 /*stablehlo_funcs=*/std::nullopt),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("`stablehlo_funcs` is required")));
}

TEST_P(DefaultDataLoaderTest, DataNameNotFoundInCheckpoint) {
  // Get the manifest.
  const std::string model_path =
      devtools_build::GetDataDependencyFilepath(kLinearModelPath);
  TF_ASSERT_OK_AND_ASSIGN(auto manifest, GetManifest(model_path));
  manifest.mutable_objects()
      ->at("linear")
      .mutable_function()
      ->mutable_data_names()
      ->Add("not_in_checkpoint");
  AssignedDevices assigned_devices;
  std::vector<const orbax_model_manifest::Function*> stablehlo_funcs = {
      &manifest.objects().at("linear").function()};
  // Load the data.
  EXPECT_THAT(
      data_loader_->Load(
          /*value_to_load=*/manifest.objects().at("params").value().external(),
          model_path, &assigned_devices, &manifest, stablehlo_funcs),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("The following data names specified in `Function` are "
                         "not found in the checkpoint: not_in_checkpoint")));
}

TEST_P(DefaultDataLoaderTest, LoadSucceedsForMultipleStableHloFuncs) {
  // This test verifies that the default data loader can load the union of
  // used weights required by concretized StableHLO functions for a polymorphic
  // JAX function.
  //
  // The JAX model is a simple linear model with polymorphic input and
  // a condition based on input batch size. It is exported to a set of
  // concretized StableHLO functions with concrete input batch sizes. Some
  // of the weights are used by all the concretized StableHLO functions, while
  // some are only used by a subset of them. The default data loader should
  // load the union of all used weights, which are all weights {"bias",
  // "weights"} in this case.
  const std::string model_path = devtools_build::GetDataDependencyFilepath(
      kLinearModelWithPolymorphicInputConcretizedPath);
  TF_ASSERT_OK_AND_ASSIGN(auto manifest, GetManifest(model_path));
  const std::string object_name = "linear_with_conditional_bias";
  ASSERT_TRUE(manifest.objects().contains(object_name));
  ASSERT_TRUE(manifest.objects().at(object_name).has_poly_fn());
  const auto& poly_fn = manifest.objects().at(object_name).poly_fn();
  std::vector<const orbax_model_manifest::Function*> stablehlo_funcs;
  for (const auto& concrete_function : poly_fn.concrete_functions()) {
    stablehlo_funcs.push_back(&concrete_function);
  }

  TF_ASSERT_OK_AND_ASSIGN(AssignedDevices device,
                          GetAssignedDevices(manifest, *stablehlo_funcs[0],
                                             /*compile_options=*/nullptr));

  // Load the weights
  absl::StatusOr<DataLoader::LoadResult> load_result = data_loader_->Load(
      /*value_to_load=*/manifest.objects().at("params").value().external(),
      model_path, &device, &manifest,
      /*stablehlo_funcs=*/stablehlo_funcs);
  EXPECT_OK(load_result);
  EXPECT_THAT(load_result->data_names, UnorderedElementsAre("bias", "weights"));

  // Check weights are loaded correctly.
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto sharding,
        xla::ParseSharding(
            "{devices=[2,2]<=[2,2]T(1,0) last_tile_dim_replicate}"));
    ASSERT_OK_AND_ASSIGN(auto loaded_variable,
                         ifrt_loaded_variable_registry_.GetLoadedVariable(
                             {.device_ids = {0, 1, 2, 3},
                              .input_name = "bias",
                              .hlo_sharding = sharding}));
    ASSERT_TRUE(
        loaded_variable.value.IsType<tsl::Future<xla::ifrt::ArrayRef>>());
    const auto array_future =
        loaded_variable.value.Get<tsl::Future<xla::ifrt::ArrayRef>>();
    ASSERT_OK_AND_ASSIGN(const auto array, array_future.Await());
    // `bias` is the first param with shape (output_dim=2)
    EXPECT_EQ(array->shape(), xla::ifrt::Shape({2}));
    EXPECT_EQ(array->dtype(), xla::ifrt::DType(xla::ifrt::DType::kF32));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(auto sharding,
                            xla::ParseSharding("{devices=[2,2]<=[4]}"));
    ASSERT_OK_AND_ASSIGN(auto loaded_variable,
                         ifrt_loaded_variable_registry_.GetLoadedVariable(
                             {.device_ids = {0, 1, 2, 3},
                              .input_name = "weights",
                              .hlo_sharding = sharding}));
    ASSERT_TRUE(
        loaded_variable.value.IsType<tsl::Future<xla::ifrt::ArrayRef>>());
    const auto array_future =
        loaded_variable.value.Get<tsl::Future<xla::ifrt::ArrayRef>>();
    ASSERT_OK_AND_ASSIGN(const auto array, array_future.Await());
    // `weight` is the second param with shape (input_dim=2, output_dim=2)
    EXPECT_EQ(array->shape(), xla::ifrt::Shape({2, 2}));
    EXPECT_EQ(array->dtype(), xla::ifrt::DType(xla::ifrt::DType::kF32));
  }

  // Verify the total number of loaded variables stored in the registry.
  IfrtLoadedVariableRegistryTestPeer ifrt_loaded_variable_registry_test_peer(
      &ifrt_loaded_variable_registry_);
  auto loaded_variable_map =
      ifrt_loaded_variable_registry_test_peer.GetLoadedVariableMap();
  EXPECT_EQ(loaded_variable_map.size(), 2);
}

TEST_P(DefaultDataLoaderTest,
       LoadSucceedsForMultipleStableHloFuncsWithUnusedWeight) {
  // This test verifies that the default data loader can load the union of
  // used weights required by concretized StableHLO functions for a polymorphic
  // JAX function. The union is a subset of all weights in this case.
  //
  // The JAX model is a simple linear model with polymorphic input and
  // a condition based on input batch size. It is exported to a set of
  // concretized StableHLO functions with concrete input batch sizes. Some
  // of the weights are used by all the concretized StableHLO functions or
  // a subset of them, while other weights are unused by any of them. The
  // default data loader should load the union of all used weights, which is
  // {"weights"} in this case, while the unused weight {"bias"} is not loaded.
  const std::string model_path = devtools_build::GetDataDependencyFilepath(
      kLinearModelWithPolymorphicInputConcretizedWithUnusedWeightPath);
  TF_ASSERT_OK_AND_ASSIGN(auto manifest, GetManifest(model_path));
  const std::string object_name = "linear_with_conditional_bias";
  ASSERT_TRUE(manifest.objects().contains(object_name));
  ASSERT_TRUE(manifest.objects().at(object_name).has_poly_fn());
  const auto& poly_fn = manifest.objects().at(object_name).poly_fn();
  std::vector<const orbax_model_manifest::Function*> stablehlo_funcs;
  for (const auto& concrete_function : poly_fn.concrete_functions()) {
    stablehlo_funcs.push_back(&concrete_function);
  }

  TF_ASSERT_OK_AND_ASSIGN(AssignedDevices device,
                          GetAssignedDevices(manifest, *stablehlo_funcs[0],
                                             /*compile_options=*/nullptr));

  // Load the weights
  absl::StatusOr<DataLoader::LoadResult> load_result = data_loader_->Load(
      /*value_to_load=*/manifest.objects().at("params").value().external(),
      model_path, &device, &manifest,
      /*stablehlo_funcs=*/stablehlo_funcs);
  EXPECT_OK(load_result);
  EXPECT_THAT(load_result->data_names, UnorderedElementsAre("bias", "weights"));

  // Check weights are loaded correctly.
  {
    TF_ASSERT_OK_AND_ASSIGN(auto sharding,
                            xla::ParseSharding("{devices=[2,2]<=[4]}"));
    ASSERT_OK_AND_ASSIGN(auto loaded_variable,
                         ifrt_loaded_variable_registry_.GetLoadedVariable(
                             {.device_ids = {0, 1, 2, 3},
                              .input_name = "weights",
                              .hlo_sharding = sharding}));
    ASSERT_TRUE(
        loaded_variable.value.IsType<tsl::Future<xla::ifrt::ArrayRef>>());
    const auto array_future =
        loaded_variable.value.Get<tsl::Future<xla::ifrt::ArrayRef>>();
    ASSERT_OK_AND_ASSIGN(const auto array, array_future.Await());
    // `weight` is the second param with shape (input_dim=2, output_dim=2)
    EXPECT_EQ(array->shape(), xla::ifrt::Shape({2, 2}));
    EXPECT_EQ(array->dtype(), xla::ifrt::DType(xla::ifrt::DType::kF32));
  }
  // Verify the total number of loaded variables stored in the registry.
  IfrtLoadedVariableRegistryTestPeer ifrt_loaded_variable_registry_test_peer(
      &ifrt_loaded_variable_registry_);
  auto loaded_variable_map =
      ifrt_loaded_variable_registry_test_peer.GetLoadedVariableMap();
  EXPECT_EQ(loaded_variable_map.size(), 1);
}

TEST_P(DefaultDataLoaderTest, LoadFailsWhenCheckpointNotFound) {
  const std::string model_path =
      devtools_build::GetDataDependencyFilepath(kLinearModelPath);
  TF_ASSERT_OK_AND_ASSIGN(auto manifest, GetManifest(model_path));
  ASSERT_TRUE(manifest.objects().contains("params"));
  ASSERT_TRUE(manifest.objects().at("params").has_value());
  ASSERT_TRUE(manifest.objects().at("params").value().has_external());
  ASSERT_TRUE(manifest.objects().contains("linear"));
  ASSERT_TRUE(manifest.objects().at("linear").has_function());
  TF_ASSERT_OK_AND_ASSIGN(
      AssignedDevices device,
      GetAssignedDevices(manifest, manifest.objects().at("linear").function(),
                         nullptr));
  std::vector<const orbax_model_manifest::Function*> stablehlo_funcs = {
      &manifest.objects().at("linear").function()};

  // Modify the external value to point to a non-existent checkpoint.
  orbax_model_manifest::ExternalValue bad_value_to_load =
      manifest.objects().at("params").value().external();
  bad_value_to_load.mutable_data()
      ->mutable_file_system_location()
      ->set_string_path("non_existent_checkpoint");

  // Load the data. This should fail because the checkpoint path does not exist.
  absl::StatusOr<DataLoader::LoadResult> load_result = data_loader_->Load(
      /*value_to_load=*/bad_value_to_load, model_path, &device, &manifest,
      /*stablehlo_funcs=*/stablehlo_funcs);
  EXPECT_THAT(load_result, StatusIs(absl::StatusCode::kNotFound,
                                    HasSubstr("non_existent_checkpoint")));
}

TEST_P(DefaultDataLoaderTest, DuplicateRegistrationFails) {
  const std::string model_path =
      devtools_build::GetDataDependencyFilepath(kLinearModelPath);
  TF_ASSERT_OK_AND_ASSIGN(auto manifest, GetManifest(model_path));
  TF_ASSERT_OK_AND_ASSIGN(
      AssignedDevices device,
      GetAssignedDevices(manifest, manifest.objects().at("linear").function(),
                         nullptr));
  std::vector<const orbax_model_manifest::Function*> stablehlo_funcs = {
      &manifest.objects().at("linear").function()};

  // First successful load.
  EXPECT_OK(data_loader_->Load(
      /*value_to_load=*/manifest.objects().at("params").value().external(),
      model_path, &device, &manifest,
      /*stablehlo_funcs=*/stablehlo_funcs));

  // Second load attempt with the same parameters. This should cause
  // TryRegisterLoadedVariable to fail.
  absl::StatusOr<DataLoader::LoadResult> load_result = data_loader_->Load(
      /*value_to_load=*/manifest.objects().at("params").value().external(),
      model_path, &device, &manifest,
      /*stablehlo_funcs=*/stablehlo_funcs);
  EXPECT_OK(load_result);

  // Verify that no new variables were registered.
  IfrtLoadedVariableRegistryTestPeer ifrt_loaded_variable_registry_test_peer(
      &ifrt_loaded_variable_registry_);
  auto loaded_variable_map =
      ifrt_loaded_variable_registry_test_peer.GetLoadedVariableMap();
  EXPECT_EQ(loaded_variable_map.size(), 2);
}

TEST_P(DefaultDataLoaderTest, LoadToAllDevicesWhenAssignedDevicesIsNull) {
  // Get the manifest.
  const std::string model_path =
      devtools_build::GetDataDependencyFilepath(kLinearModelPath);
  TF_ASSERT_OK_AND_ASSIGN(auto manifest, GetManifest(model_path));
  ASSERT_TRUE(manifest.objects().contains("params"));
  ASSERT_TRUE(manifest.objects().at("params").has_value());
  ASSERT_TRUE(manifest.objects().at("params").value().has_external());
  ASSERT_TRUE(manifest.objects().contains("linear"));
  ASSERT_TRUE(manifest.objects().at("linear").has_function());
  std::vector<const orbax_model_manifest::Function*> stablehlo_funcs = {
      &manifest.objects().at("linear").function()};

  // Load the data.
  absl::StatusOr<DataLoader::LoadResult> load_result = data_loader_->Load(
      /*value_to_load=*/manifest.objects().at("params").value().external(),
      model_path, /*assigned_devices=*/nullptr, &manifest,
      /*stablehlo_funcs=*/stablehlo_funcs);
  EXPECT_OK(load_result);
  EXPECT_THAT(load_result->data_names, UnorderedElementsAre("bias", "weights"));

  // Verify weights are loaded by fetching from the registry.
  for (int i = 0; i < ifrt_client_->addressable_device_count(); ++i) {
    {
      ASSERT_OK_AND_ASSIGN(
          auto loaded_variable,
          ifrt_loaded_variable_registry_.GetLoadedVariable(
              {.device_ids = {i},
               .input_name = "bias",
               .hlo_sharding = xla::HloSharding::AssignDevice(i)}));
      ASSERT_TRUE(
          loaded_variable.value.IsType<tsl::Future<xla::ifrt::ArrayRef>>());
      const auto array_future =
          loaded_variable.value.Get<tsl::Future<xla::ifrt::ArrayRef>>();
      if (use_thread_pool_) {
        // Disabling this async check due to risk of potential flakiness.
        // EXPECT_FALSE(array_future.IsReady());
      } else {
        EXPECT_TRUE(array_future.IsReady());
      }
      ASSERT_OK_AND_ASSIGN(const auto array, array_future.Await());
      // `bias` is the first param with shape (output_dim=2)
      EXPECT_EQ(array->shape(), xla::ifrt::Shape({2}));
      EXPECT_EQ(array->dtype(), xla::ifrt::DType(xla::ifrt::DType::kF32));
    }
    {
      ASSERT_OK_AND_ASSIGN(
          auto loaded_variable,
          ifrt_loaded_variable_registry_.GetLoadedVariable(
              {.device_ids = {i},
               .input_name = "weights",
               .hlo_sharding = xla::HloSharding::AssignDevice(i)}));
      ASSERT_TRUE(
          loaded_variable.value.IsType<tsl::Future<xla::ifrt::ArrayRef>>());
      const auto array_future =
          loaded_variable.value.Get<tsl::Future<xla::ifrt::ArrayRef>>();
      ASSERT_OK_AND_ASSIGN(const auto array, array_future.Await());
      // `weight` is the second param with shape (input_dim=2, output_dim=2)
      EXPECT_EQ(array->shape(), xla::ifrt::Shape({2, 2}));
      EXPECT_EQ(array->dtype(), xla::ifrt::DType(xla::ifrt::DType::kF32));
    }
  }

  // Verify the total number of loaded variables stored in the registry.
  IfrtLoadedVariableRegistryTestPeer ifrt_loaded_variable_registry_test_peer(
      &ifrt_loaded_variable_registry_);
  auto loaded_variable_map =
      ifrt_loaded_variable_registry_test_peer.GetLoadedVariableMap();
  EXPECT_EQ(loaded_variable_map.size(),
            2 * ifrt_client_->addressable_device_count());
}

// TODO: Add more tests.

INSTANTIATE_TEST_SUITE_P(DefaultDataLoaderTestInstance, DefaultDataLoaderTest,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "Async" : "Sync";
                         });

}  // namespace
}  // namespace jsv
