#include "third_party/neptune/runtime/interpreter/manifest_utils.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "devtools/build/runtime/get_runfiles_dir.h"
#include "third_party/neptune/runtime/interpreter/test_util.h"
#include "net/proto2/contrib/parse_proto/parse_text_proto.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/types/span.h"
#include "third_party/neptune/runtime/core/constants.h"
#include "third_party/neptune/runtime/core/flat_signature.h"
#include "third_party/neptune/runtime/core/stablehlo_utils.h"
#include "third_party/neptune/runtime/core/type.h"
#include "third_party/neptune/runtime/core/value_matcher.h"
#include "third_party/py/orbax/experimental/model/core/protos/manifest.proto.h"
#include "third_party/py/orbax/experimental/model/core/protos/type.proto.h"
#include "third_party/py/orbax/export/protos/oex_orchestration.proto.h"
#include "third_party/tensorflow/compiler/xla/hlo/ir/hlo_sharding.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/dtype.h"
#include "third_party/tensorflow/compiler/xla/tests/test_utils.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/env.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/status.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"
#include "third_party/tensorflow/compiler/xla/xla_data.proto.h"
#include "third_party/tensorflow/tsl/platform/path.h"

namespace jsv {
namespace {

using ::jsv::kOrbaxModelManifestVersionTxt;
using ::jsv::test_util::MetaArgSpecsAre;
using ::jsv::type::Dict;
using ::jsv::type::StringTypePairs;
using ::jsv::type::Tuple;
using ::jsv::type::Type;
using ::orbax_model_manifest::UnstructuredData::kFileSystemLocation;
using ::orbax_model_manifest::UnstructuredData::kInlinedString;
using ::proto2::contrib::parse_proto::ParseTextProtoOrDie;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::EqualsProto;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Matcher;
using ::testing::Pair;
using ::testing::status::StatusIs;
using ::xla::ifrt::DType;

using OpShardingRoundTripTest = ::testing::TestWithParam<absl::string_view>;
using GetInputFlatSignatureNonArgsKwargsPatternTest =
    ::testing::TestWithParam<orbax_model_type::Type>;

constexpr absl::string_view kInputListAndDictTypeProtoPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/testdata/"
    "type/input_list_and_dict.textpb";
constexpr absl::string_view kInputTupleTypeProtoPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/testdata/"
    "type/input_tuple.textpb";
constexpr absl::string_view kOutputTupleTypeProtoPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/testdata/"
    "type/output_tuple.textpb";
constexpr absl::string_view kInputDictTypeProtoPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/testdata/"
    "type/input_dict.textpb";
constexpr absl::string_view kOutputDictTypeProtoPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/testdata/"
    "type/output_dict.textpb";
constexpr absl::string_view kInputTupleWithOffsetTypeProtoPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/testdata/"
    "type/input_tuple_with_offset.textpb";
constexpr absl::string_view kInputKwOnlyTypeProtoPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/testdata/"
    "type/input_kw_only.textpb";
constexpr absl::string_view kInputStringTypePairsTypeProtoPath =
    "google3/learning/infra/mira/experimental/orbax_model/interpreter/testdata/"
    "type/input_string_type_pairs.textpb";

orbax_model_type::Type MakeLeafType(const orbax_model_type::DType& dtype,
                                    absl::Span<const uint64_t> shape) {
  orbax_model_type::Type leaf;
  leaf.mutable_leaf()->mutable_tensor_type()->set_dtype(dtype);
  orbax_model_type::ShapeWithKnownRank* shape_proto =
      leaf.mutable_leaf()
          ->mutable_tensor_type()
          ->mutable_shape()
          ->mutable_shape_with_known_rank();
  for (const uint64_t size : shape) {
    shape_proto->add_dimension_sizes()->set_size(size);
  }
  return leaf;
}

// Gets the `jsv::Type` from a `orbax_model_type::Type`text proto file.
absl::StatusOr<orbax_model_type::Type> GetTypeFromTextProto(
    absl::string_view type_text_proto_path) {
  std::string type_text;
  TF_CHECK_OK(tsl::ReadFileToString(
      tsl::Env::Default(),
      devtools_build::GetDataDependencyFilepath(type_text_proto_path),
      &type_text));
  return xla::ParseTextProto<orbax_model_type::Type>(type_text);
}

// Generates the `orbax_model_type::Type`s that do not use the args/kwargs
// pattern.
std::vector<orbax_model_type::Type> GetNonArgsKwargsPatternTypes() {
  std::vector<orbax_model_type::Type> types;
  // (,)
  {
    orbax_model_type::Type type;
    type.mutable_tuple();
    types.push_back(type);
  }
  // {}
  {
    orbax_model_type::Type type;
    type.mutable_dict();
    types.push_back(type);
  }
  // ({},)
  {
    orbax_model_type::Type type;
    type.mutable_tuple()->add_elements()->mutable_dict();
    types.push_back(type);
  }
  // ((,), (,))
  {
    orbax_model_type::Type type;
    type.mutable_tuple()->add_elements()->mutable_tuple();
    type.mutable_tuple()->add_elements()->mutable_tuple();
    types.push_back(type);
  }
  // ({}, (,))
  {
    orbax_model_type::Type type;
    type.mutable_tuple()->add_elements()->mutable_dict();
    type.mutable_tuple()->add_elements()->mutable_tuple();
    types.push_back(type);
  }
  // [[], []]
  {
    orbax_model_type::Type type;
    type.mutable_list()->add_elements()->mutable_list();
    type.mutable_list()->add_elements()->mutable_list();
    types.push_back(type);
  }
  // [(,), (,)]
  {
    orbax_model_type::Type type;
    type.mutable_list()->add_elements()->mutable_tuple();
    type.mutable_list()->add_elements()->mutable_tuple();
    types.push_back(type);
  }
  // ({(,), (,), {}},)
  {
    orbax_model_type::Type type;
    type.mutable_list()->add_elements()->mutable_tuple();
    type.mutable_list()->add_elements()->mutable_tuple();
    type.mutable_tuple()->add_elements()->mutable_dict();
    types.push_back(type);
  }
  return types;
}

// Makes a `orbax_model_type::TensorType` with the given dtype, shape, and
// sharding.
orbax_model_type::TensorType MakeTensorType(
    orbax_model_type::DType dtype, absl::Span<const int64_t> shape,
    xla::OpSharding sharding = xla::OpSharding()) {
  orbax_model_type::TensorType tensor_type;
  tensor_type.set_dtype(dtype);
  orbax_model_type::ShapeWithKnownRank* shape_proto =
      tensor_type.mutable_shape()->mutable_shape_with_known_rank();
  for (int64_t dim : shape) {
    shape_proto->add_dimension_sizes()->set_size(dim);
  }
  *tensor_type.mutable_sharding() = sharding;
  return tensor_type;
}

TEST_P(OpShardingRoundTripTest, RoundTrip) {
  xla::OpSharding orbax_sharding = ParseTextProtoOrDie(GetParam());

  xla::OpSharding expected_xla_op_sharding = ParseTextProtoOrDie(GetParam());

  xla::OpSharding xla_sharding = orbax_sharding;

  EXPECT_THAT(xla_sharding, EqualsProto(expected_xla_op_sharding));
}
INSTANTIATE_TEST_SUITE_P(OpShardingRoundTripTest, OpShardingRoundTripTest,
                         testing::ValuesIn<absl::string_view>({
                             R"pb(
                               type: MAXIMAL
                               tile_shape: {}
                             )pb",
                             R"pb(
                               type: REPLICATED
                             )pb",
                             R"pb(
                               type: OTHER
                               tile_assignment_devices: 0
                               tile_assignment_devices: 1
                               tile_assignment_dimensions: 2
                               tile_shape: {}
                             )pb",
                             // A sharding that sets last two fields.
                             R"pb(
                               type: OTHER
                               tile_assignment_devices: 0
                               tile_assignment_devices: 1
                               tile_assignment_dimensions: 2
                               tile_shape: {}
                               shard_group_id: 1234567
                               shard_group_type: LIKE
                             )pb",
                         }));

TEST(ManifestUtilsTest, CheckMimeTypeSuccess) {
  orbax_model_manifest::UnstructuredData unstructured_data;
  unstructured_data.set_mime_type("type0");
  EXPECT_OK(CheckMimeType(unstructured_data, "type0", "An object"));
}

TEST(ManifestUtilsTest, CheckMimeTypeFailure) {
  orbax_model_manifest::UnstructuredData unstructured_data;
  unstructured_data.set_mime_type("type0");
  EXPECT_THAT(CheckMimeType(unstructured_data, "type1", "An object"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("An object: only mime type \"type1\" is "
                                 "supported, but got mime type \"type0\"")));
}

TEST(ManifestUtilsTest, CheckMimeTypeUnset) {
  orbax_model_manifest::UnstructuredData unstructured_data;
  EXPECT_THAT(CheckMimeType(unstructured_data, "type1", "An object"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("An object: mime type is not set.")));
}

TEST(ManifestUtilsTest, CheckDataTypeSuccess) {
  orbax_model_manifest::UnstructuredData unstructured_data;
  unstructured_data.mutable_inlined_string();
  EXPECT_OK(CheckDataType(unstructured_data, kInlinedString, "An object"));
}

TEST(ManifestUtilsTest, CheckDataTypeFailure) {
  orbax_model_manifest::UnstructuredData unstructured_data;
  unstructured_data.mutable_inlined_string();
  EXPECT_THAT(
      CheckDataType(unstructured_data, kFileSystemLocation, "An object"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("An object: only data type \"file_system_location\" is "
                    "supported, but got data type \"inlined_string\"")));
}

TEST(ManifestUtilsTest, CheckDataTypeUnset) {
  orbax_model_manifest::UnstructuredData unstructured_data;
  EXPECT_THAT(
      CheckDataType(unstructured_data, kFileSystemLocation, "An object"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("An object: only data type \"file_system_location\" is "
                    "supported, but got data type \"DATA_NOT_SET\"")));
}

TEST(ManifestUtilsTest, CheckMimeAndDataTypeSuccess) {
  orbax_model_manifest::UnstructuredData unstructured_data;
  unstructured_data.set_mime_type("type0");
  unstructured_data.mutable_inlined_string();
  EXPECT_OK(CheckMimeAndDataType(unstructured_data, "type0", kInlinedString,
                                 "An object"));
}

TEST(ManifestUtilsTest, CheckMimeAndDataTypeDataTypeMismatch) {
  orbax_model_manifest::UnstructuredData unstructured_data;
  unstructured_data.set_mime_type("type0");
  unstructured_data.mutable_inlined_string();
  EXPECT_THAT(CheckMimeAndDataType(unstructured_data, "type0",
                                   kFileSystemLocation, "An object"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(FlattenTypeTest, ListType) {
  orbax_model_type::Type leaf = MakeLeafType(orbax_model_type::DType::f32, {1});

  orbax_model_type::Type type;
  *type.mutable_list()->add_elements() = leaf;

  std::vector<orbax_model_type::LeafType> flatten_types;
  EXPECT_OK(FlattenType(type, flatten_types));
}

TEST(FlattenTypeTest, DictType) {
  orbax_model_type::Type leaf_1 =
      MakeLeafType(orbax_model_type::DType::f32, {1});
  orbax_model_type::Type leaf_2 =
      MakeLeafType(orbax_model_type::DType::si32, {2});

  orbax_model_type::Type type;
  proto2::Map<std::string, orbax_model_type::Type>* string_to_type =
      type.mutable_dict()->mutable_string_to_type();
  (*string_to_type)["b"] = leaf_1;
  (*string_to_type)["a"] = leaf_2;
  std::vector<orbax_model_type::LeafType> expected_flat_types = {leaf_2.leaf(),
                                                                 leaf_1.leaf()};
  std::vector<orbax_model_type::LeafType> flat_types;
  EXPECT_OK(FlattenType(type, flat_types));
  ASSERT_EQ(flat_types.size(), expected_flat_types.size());
  for (int i = 0; i < flat_types.size(); ++i) {
    EXPECT_THAT(flat_types[i], EqualsProto(expected_flat_types[i]));
  }
}

TEST(FlatSignatureTest, GetInputFlatSignatureTuple) {
  TF_ASSERT_OK_AND_ASSIGN(orbax_model_type::Type type,
                          GetTypeFromTextProto(kInputTupleTypeProtoPath));
  TF_ASSERT_OK_AND_ASSIGN(FlatSignature flat_signature,
                          GetInputFlatSignature(type, /*offset=*/0));
  const std::vector<MetaArgSpec> expected_specs = {
      MetaArgSpec{.spec = ArgSpec{.dtype = DType(DType::kF32),
                                  .shape = ArgShape({2, 2}),
                                  .sharding = xla::HloSharding::Replicate()}},
      MetaArgSpec{
          .spec = ArgSpec{.dtype = DType(DType::kF32),
                          .shape = ArgShape({2, 4}),
                          .sharding = xla::HloSharding::Replicate()},
      }};
  EXPECT_THAT(flat_signature, MetaArgSpecsAre(expected_specs));
}

TEST(FlatSignatureTest, GetInputFlatSignatureStringTypePairs) {
  TF_ASSERT_OK_AND_ASSIGN(
      orbax_model_type::Type type,
      GetTypeFromTextProto(kInputStringTypePairsTypeProtoPath));
  TF_ASSERT_OK_AND_ASSIGN(FlatSignature flat_signature,
                          GetInputFlatSignature(type, /*offset=*/0));
  const std::vector<MetaArgSpec> expected_specs = {
      MetaArgSpec{.spec = ArgSpec{.dtype = DType(DType::kF32),
                                  .shape = ArgShape({2, 2}),
                                  .sharding = xla::HloSharding::Replicate()}},
      MetaArgSpec{
          .spec = ArgSpec{.dtype = DType(DType::kF32),
                          .shape = ArgShape({2, 4}),
                          .sharding = xla::HloSharding::Replicate()},
      },
      MetaArgSpec{.spec = ArgSpec{.dtype = DType(DType::kF32),
                                  .shape = ArgShape({2, 2}),
                                  .sharding = xla::HloSharding::Replicate()}},
      MetaArgSpec{
          .spec = ArgSpec{.dtype = DType(DType::kF32),
                          .shape = ArgShape({2, 4}),
                          .sharding = xla::HloSharding::Replicate()},
      }};
  EXPECT_THAT(flat_signature, MetaArgSpecsAre(expected_specs));
}

TEST(FlatSignatureTest, GetInputFlatSignatureListAndDict) {
  TF_ASSERT_OK_AND_ASSIGN(orbax_model_type::Type type,
                          GetTypeFromTextProto(kInputListAndDictTypeProtoPath));
  TF_ASSERT_OK_AND_ASSIGN(FlatSignature flat_signature,
                          GetInputFlatSignature(type, /*offset=*/0));
  const std::vector<MetaArgSpec> expected_specs = {
      MetaArgSpec{.spec = ArgSpec{.dtype = DType(DType::kU16),
                                  .shape = ArgShape({5, 3}),
                                  .sharding = xla::HloSharding::Replicate()}},
      MetaArgSpec{
          .spec = ArgSpec{.dtype = DType(DType::kS32),
                          .shape = ArgShape({7, 2}),
                          .sharding = xla::HloSharding::Replicate()},
      }};
  EXPECT_THAT(flat_signature, MetaArgSpecsAre(expected_specs));
}

TEST(FlatSignatureTest, GetInputFlatSignatureDict) {
  TF_ASSERT_OK_AND_ASSIGN(orbax_model_type::Type type,
                          GetTypeFromTextProto(kInputDictTypeProtoPath));

  TF_ASSERT_OK_AND_ASSIGN(FlatSignature flat_signature,
                          GetInputFlatSignature(type, /*offset=*/0));
  const std::vector<MetaArgSpec> expected_specs = {
      MetaArgSpec{.spec = ArgSpec{.dtype = DType(DType::kF32),
                                  .shape = ArgShape({2, 2}),
                                  .sharding = xla::HloSharding::Replicate()}},
      MetaArgSpec{.spec = ArgSpec{.dtype = DType(DType::kF32),
                                  .shape = ArgShape({2, 4}),
                                  .sharding = xla::HloSharding::Replicate()}}};
  EXPECT_THAT(flat_signature, MetaArgSpecsAre(expected_specs));
}

TEST(FlatSignatureTest, GetInputFlatSignatureWithOffset) {
  TF_ASSERT_OK_AND_ASSIGN(
      orbax_model_type::Type type,
      GetTypeFromTextProto(kInputTupleWithOffsetTypeProtoPath));

  TF_ASSERT_OK_AND_ASSIGN(FlatSignature flat_signature,
                          GetInputFlatSignature(type, /*offset=*/1));
  MetaArgSpec expected_spec{
      .spec = ArgSpec{.dtype = DType(DType::kF32),
                      .shape = ArgShape({4, 2}),
                      .sharding = xla::HloSharding::IotaTile({2, 2})}};
  ASSERT_EQ(flat_signature.specs().size(), 1);
  EXPECT_TRUE(flat_signature.specs()[0].EqualExceptDefaultValue(expected_spec));
}

TEST(FlatSignatureTest, GetInputFlatSignatureKwOnly) {
  TF_ASSERT_OK_AND_ASSIGN(orbax_model_type::Type type,
                          GetTypeFromTextProto(kInputKwOnlyTypeProtoPath));

  TF_ASSERT_OK_AND_ASSIGN(FlatSignature flat_signature,
                          GetInputFlatSignature(type));
  const std::vector<MetaArgSpec> expected_specs = {
      MetaArgSpec{.spec = ArgSpec{.dtype = DType(DType::kF32),
                                  .shape = ArgShape({2, 2}),
                                  .sharding = xla::HloSharding::Replicate()}},
      MetaArgSpec{.spec = ArgSpec{.dtype = DType(DType::kF32),
                                  .shape = ArgShape({2, 4}),
                                  .sharding = xla::HloSharding::Replicate()}}};
  EXPECT_THAT(flat_signature, MetaArgSpecsAre(expected_specs));
}

TEST(FlatSignatureTest, GetInputFlatSignatureOffsetOutOfRange) {
  TF_ASSERT_OK_AND_ASSIGN(orbax_model_type::Type type,
                          GetTypeFromTextProto(kInputKwOnlyTypeProtoPath));

  EXPECT_THAT(GetInputFlatSignature(type, /*offset=*/1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Offset 1 is out of range")));
}

TEST_P(GetInputFlatSignatureNonArgsKwargsPatternTest, Fails) {
  orbax_model_type::Type type;
  // The pattern is ((,),)
  type.mutable_tuple()->add_elements()->mutable_tuple();

  EXPECT_THAT(GetInputFlatSignature(type),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("args-kwargs pattern")));
}

INSTANTIATE_TEST_SUITE_P(GetInputFlatSignatureNonArgsKwargsPatternTest,
                         GetInputFlatSignatureNonArgsKwargsPatternTest,
                         testing::ValuesIn(GetNonArgsKwargsPatternTypes()));

TEST(FlatSignatureTest, GetOutputFlatSignatureTuple) {
  TF_ASSERT_OK_AND_ASSIGN(orbax_model_type::Type type,
                          GetTypeFromTextProto(kOutputTupleTypeProtoPath));

  TF_ASSERT_OK_AND_ASSIGN(FlatSignature flat_signature,
                          GetOutputFlatSignature(type));
  const std::vector<MetaArgSpec> expected_specs = {
      MetaArgSpec{.spec = ArgSpec{.dtype = DType(DType::kF32),
                                  .shape = ArgShape({2, 2}),
                                  .sharding = xla::HloSharding::Replicate()}},
      MetaArgSpec{.spec = ArgSpec{.dtype = DType(DType::kF32),
                                  .shape = ArgShape({2, 4}),
                                  .sharding = xla::HloSharding::Replicate()}}};
  EXPECT_THAT(flat_signature, MetaArgSpecsAre(expected_specs));
}

TEST(FlatSignatureTest, GetOutputFlatSignatureDict) {
  TF_ASSERT_OK_AND_ASSIGN(orbax_model_type::Type type,
                          GetTypeFromTextProto(kOutputDictTypeProtoPath));

  TF_ASSERT_OK_AND_ASSIGN(FlatSignature flat_signature,
                          GetOutputFlatSignature(type));
  const std::vector<MetaArgSpec> expected_specs = {
      MetaArgSpec{.spec = ArgSpec{.dtype = DType(DType::kF32),
                                  .shape = ArgShape({2, 2}),
                                  .sharding = xla::HloSharding::Replicate()}},
      MetaArgSpec{.spec = ArgSpec{.dtype = DType(DType::kF32),
                                  .shape = ArgShape({2, 4}),
                                  .sharding = xla::HloSharding::Replicate()}}};
  EXPECT_THAT(flat_signature, MetaArgSpecsAre(expected_specs));
}

TEST(FlatSignatureTest, GetOutputFlatSignatureUnsupportedType) {
  orbax_model_type::Type leaf = MakeLeafType(orbax_model_type::DType::f32, {1});
  orbax_model_type::StringTypePair pair;
  pair.set_fst("a");
  *pair.mutable_snd() = leaf;

  orbax_model_type::Type type;
  *type.mutable_string_type_pairs()->add_elements() = pair;

  EXPECT_THAT(GetOutputFlatSignature(type),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("Unsupported type"),
                             HasSubstr("string_type_pairs"))));
}

TEST(FlatSignatureTest, GetFlatSignatureFromNamedSignature) {
  const std::string kOtherSharding = R"pb(
    type: OTHER
    tile_assignment_dimensions: 2
    tile_assignment_dimensions: 1
    iota_reshape_dims: 2
    iota_transpose_perm: 0
  )pb";

  orbax_export_orchestration::NamedSignature named_signature;
  {
    auto* input = named_signature.mutable_inputs()->add_named_tensor_types();
    input->set_name("a");
    *input->mutable_tensor_type() =
        MakeTensorType(orbax_model_type::DType::f32, {2, 2});
  }
  {
    auto* input = named_signature.mutable_inputs()->add_named_tensor_types();
    input->set_name("c");
    *input->mutable_tensor_type() =
        MakeTensorType(orbax_model_type::DType::si32, {4, 6},
                       *xla::ParseTextProto<xla::OpSharding>(kOtherSharding));
  }
  TF_ASSERT_OK_AND_ASSIGN(
      FlatSignature flat_signature,
      GetFlatSignatureFromNamedSignature(named_signature.inputs()));

  xla::OpSharding other_sharding =
      *xla::ParseTextProto<xla::OpSharding>(kOtherSharding);
  TF_ASSERT_OK_AND_ASSIGN(xla::HloSharding hlo_sharding,
                          xla::HloSharding::FromProto(other_sharding));

  const std::vector<MetaArgSpec> expected_specs = {
      MetaArgSpec{.spec = ArgSpec{.dtype = DType(DType::kF32),
                                  .shape = ArgShape({2, 2}),
                                  .sharding = xla::HloSharding::Replicate()},
                  .name = "a"},
      MetaArgSpec{.spec = ArgSpec{.dtype = DType(DType::kS32),
                                  .shape = ArgShape({4, 6}),
                                  .sharding = hlo_sharding},
                  .name = "c"}};
  EXPECT_THAT(flat_signature, MetaArgSpecsAre(expected_specs));
}

TEST(FlatSignatureTest, EmptyNamedSignature) {
  orbax_export_orchestration::NamedSignature named_signature;
  EXPECT_THAT(GetFlatSignatureFromNamedSignature(named_signature.inputs()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Empty named signature")));
}

TEST(FlatSignatureTest, EmptyNameInNamedSignature) {
  orbax_export_orchestration::NamedSignature named_signature;
  auto* input = named_signature.mutable_inputs()->add_named_tensor_types();
  input->set_name("");
  *input->mutable_tensor_type() =
      MakeTensorType(orbax_model_type::DType::f32, {2, 2});
  EXPECT_THAT(GetFlatSignatureFromNamedSignature(named_signature.inputs()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Empty name in named signature")));
}

TEST(GetAssignedDevicesTest, NoExplicitDeviceAssignment) {
  orbax_model_manifest::Manifest manifest;
  orbax_model_manifest::Function stable_hlo_func;
  stable_hlo_func.mutable_signature()
      ->mutable_input()
      ->mutable_leaf()
      ->mutable_tensor_type();
  orbax_model_manifest::TopLevelObject top_level_object;
  *top_level_object.mutable_function() = stable_hlo_func;
  manifest.mutable_objects()->insert({"function", top_level_object});
  TF_ASSERT_OK_AND_ASSIGN(AssignedDevices assigned_devices,
                          GetAssignedDevices(manifest, stable_hlo_func,
                                             /*compile_options=*/nullptr));
  EXPECT_EQ(assigned_devices.num_replicas, 1);
  EXPECT_EQ(assigned_devices.num_cores_per_replica, 1);
}

TEST(GetAssignedDevicesTest, IgnoreDeviceAssignmentInManifest) {
  orbax_model_manifest::Manifest manifest;
  *manifest.mutable_device_assignment_by_coords()->add_devices() =
      ParseTextProtoOrDie(
          R"pb(
            id: 0 coords: 0 coords: 0 core_on_chip: 1
          )pb");
  *manifest.mutable_device_assignment_by_coords()->add_devices() =
      ParseTextProtoOrDie(
          R"pb(
            id: 1 coords: 0 coords: 1 core_on_chip: 1
          )pb");
  xla::CompileOptionsProto compile_options;
  compile_options.mutable_executable_build_options()->set_num_replicas(1);
  compile_options.mutable_executable_build_options()->set_num_partitions(2);
  TF_ASSERT_OK_AND_ASSIGN(
      AssignedDevices assigned_devices,
      GetAssignedDevices(manifest, /*stable_hlo_func=*/{}, &compile_options,
                         /*ignore_device_assignment_in_manifest=*/true));
  EXPECT_FALSE(assigned_devices.device_assignment.has_value());
}

TEST(GetAssignedDevicesTest, HasExplicitDeviceAssignment) {
  orbax_model_manifest::Manifest manifest;
  *manifest.mutable_device_assignment_by_coords()->add_devices() =
      ParseTextProtoOrDie(
          R"pb(
            id: 0 coords: 0 coords: 0 core_on_chip: 1
          )pb");
  *manifest.mutable_device_assignment_by_coords()->add_devices() =
      ParseTextProtoOrDie(
          R"pb(
            id: 1 coords: 0 coords: 1 core_on_chip: 1
          )pb");
  xla::CompileOptionsProto compile_options;
  compile_options.mutable_executable_build_options()->set_num_replicas(1);
  compile_options.mutable_executable_build_options()->set_num_partitions(2);
  TF_ASSERT_OK_AND_ASSIGN(
      AssignedDevices assigned_devices,
      GetAssignedDevices(manifest, /*stable_hlo_func=*/{}, &compile_options));
  EXPECT_EQ(assigned_devices.num_replicas, 1);
  EXPECT_EQ(assigned_devices.num_cores_per_replica, 2);
  EXPECT_EQ(assigned_devices.device_assignment,
            std::vector<int>({0, 0, 1, 0, 1, 1}));
}

// TODO: Add test cases using exported models for
// `GetInputNamesTest` and `GetOutputNamesTest`.

TEST(GetInputNamesTest, Success) {
  orbax_export_orchestration::Pipelines pipelines;
  auto* p1_inputs = (*pipelines.mutable_name_to_pipeline())["p1"]
                        .mutable_named_signature()
                        ->mutable_inputs();
  auto* t1 = p1_inputs->add_named_tensor_types();
  t1->set_name("a");
  t1->mutable_tensor_type()->set_dtype(orbax_model_type::DType::f32);
  t1->mutable_tensor_type()
      ->mutable_shape()
      ->mutable_shape_with_known_rank()
      ->add_dimension_sizes()
      ->set_size(1);
  auto* t2 = p1_inputs->add_named_tensor_types();
  t2->set_name("b");
  t2->mutable_tensor_type()->set_dtype(orbax_model_type::DType::f32);
  t2->mutable_tensor_type()
      ->mutable_shape()
      ->mutable_shape_with_known_rank()
      ->add_dimension_sizes()
      ->set_size(1);
  const std::string path =
      tsl::io::JoinPath(testing::TempDir(), "oex_orchestration.pb");
  TF_CHECK_OK(tsl::WriteBinaryProto(tsl::Env::Default(), path, pipelines));
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::string> input_names,
                          GetInputNames(path, "p1"));
  EXPECT_THAT(input_names, ElementsAre("a", "b"));
}

TEST(GetInputNamesTest, FileNotExist) {
  EXPECT_THAT(GetInputNames("file_not_exist", "p1"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(GetInputNamesTest, PipelineNotFound) {
  orbax_export_orchestration::Pipelines pipelines;
  (*pipelines.mutable_name_to_pipeline())["p1"];
  const std::string path =
      tsl::io::JoinPath(testing::TempDir(), "oex_orchestration.pb");
  TF_CHECK_OK(tsl::WriteBinaryProto(tsl::Env::Default(), path, pipelines));
  EXPECT_THAT(
      GetInputNames(path, "p2"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Pipeline \"p2\" not found. Available pipelines: p1")));
}

TEST(GetInputNamesTest, NoNamedSignature) {
  orbax_export_orchestration::Pipelines pipelines;
  (*pipelines.mutable_name_to_pipeline())["p1"];
  const std::string path =
      tsl::io::JoinPath(testing::TempDir(), "oex_orchestration.pb");
  TF_CHECK_OK(tsl::WriteBinaryProto(tsl::Env::Default(), path, pipelines));
  EXPECT_THAT(GetInputNames(path, "p1"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Pipeline does not have a named signature.")));
}

TEST(GetInputNamesTest, NoInputsInNamedSignature) {
  orbax_export_orchestration::Pipelines pipelines;
  (*pipelines.mutable_name_to_pipeline())["p1"].mutable_named_signature();
  const std::string path =
      tsl::io::JoinPath(testing::TempDir(), "oex_orchestration.pb");
  TF_CHECK_OK(tsl::WriteBinaryProto(tsl::Env::Default(), path, pipelines));
  EXPECT_THAT(GetInputNames(path, "p1"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Named signature does not have inputs.")));
}

TEST(GetInputNamesTest, EmptyInputName) {
  orbax_export_orchestration::Pipelines pipelines;
  auto* t1 = (*pipelines.mutable_name_to_pipeline())["p1"]
                 .mutable_named_signature()
                 ->mutable_inputs()
                 ->add_named_tensor_types();
  t1->set_name("");
  t1->mutable_tensor_type()->set_dtype(orbax_model_type::DType::f32);
  t1->mutable_tensor_type()
      ->mutable_shape()
      ->mutable_shape_with_known_rank()
      ->add_dimension_sizes()
      ->set_size(1);
  const std::string path =
      tsl::io::JoinPath(testing::TempDir(), "oex_orchestration.pb");
  TF_CHECK_OK(tsl::WriteBinaryProto(tsl::Env::Default(), path, pipelines));
  EXPECT_THAT(GetInputNames(path, "p1"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Empty name in named signature")));
}

TEST(GetOutputNamesTest, Success) {
  orbax_export_orchestration::Pipelines pipelines;
  auto* p1_outputs = (*pipelines.mutable_name_to_pipeline())["p1"]
                         .mutable_named_signature()
                         ->mutable_outputs();
  auto* t1 = p1_outputs->add_named_tensor_types();
  t1->set_name("a");
  t1->mutable_tensor_type()->set_dtype(orbax_model_type::DType::f32);
  t1->mutable_tensor_type()
      ->mutable_shape()
      ->mutable_shape_with_known_rank()
      ->add_dimension_sizes()
      ->set_size(1);
  auto* t2 = p1_outputs->add_named_tensor_types();
  t2->set_name("b");
  t2->mutable_tensor_type()->set_dtype(orbax_model_type::DType::f32);
  t2->mutable_tensor_type()
      ->mutable_shape()
      ->mutable_shape_with_known_rank()
      ->add_dimension_sizes()
      ->set_size(1);
  const std::string path =
      tsl::io::JoinPath(testing::TempDir(), "oex_orchestration.pb");
  TF_CHECK_OK(tsl::WriteBinaryProto(tsl::Env::Default(), path, pipelines));
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::string> output_names,
                          GetOutputNames(path, "p1"));
  EXPECT_THAT(output_names, ElementsAre("a", "b"));
}

TEST(GetOutputNamesTest, FileNotExist) {
  EXPECT_THAT(GetOutputNames("file_not_exist", "p1"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(GetOutputNamesTest, PipelineNotFound) {
  orbax_export_orchestration::Pipelines pipelines;
  (*pipelines.mutable_name_to_pipeline())["p1"];
  const std::string path =
      tsl::io::JoinPath(testing::TempDir(), "oex_orchestration.pb");
  TF_CHECK_OK(tsl::WriteBinaryProto(tsl::Env::Default(), path, pipelines));
  EXPECT_THAT(
      GetOutputNames(path, "p2"),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Pipeline \"p2\" not found. Available pipelines: p1")));
}

TEST(GetOutputNamesTest, NoNamedSignature) {
  orbax_export_orchestration::Pipelines pipelines;
  (*pipelines.mutable_name_to_pipeline())["p1"];
  const std::string path =
      tsl::io::JoinPath(testing::TempDir(), "oex_orchestration.pb");
  TF_CHECK_OK(tsl::WriteBinaryProto(tsl::Env::Default(), path, pipelines));
  EXPECT_THAT(GetOutputNames(path, "p1"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Pipeline does not have a named signature.")));
}

TEST(GetOutputNamesTest, NoOutputsInNamedSignature) {
  orbax_export_orchestration::Pipelines pipelines;
  (*pipelines.mutable_name_to_pipeline())["p1"].mutable_named_signature();
  const std::string path =
      tsl::io::JoinPath(testing::TempDir(), "oex_orchestration.pb");
  TF_CHECK_OK(tsl::WriteBinaryProto(tsl::Env::Default(), path, pipelines));
  EXPECT_THAT(GetOutputNames(path, "p1"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Named signature does not have outputs.")));
}

TEST(GetOutputNamesTest, EmptyOutputName) {
  orbax_export_orchestration::Pipelines pipelines;
  auto* t1 = (*pipelines.mutable_name_to_pipeline())["p1"]
                 .mutable_named_signature()
                 ->mutable_outputs()
                 ->add_named_tensor_types();
  t1->set_name("");
  t1->mutable_tensor_type()->set_dtype(orbax_model_type::DType::f32);
  t1->mutable_tensor_type()
      ->mutable_shape()
      ->mutable_shape_with_known_rank()
      ->add_dimension_sizes()
      ->set_size(1);
  const std::string path =
      tsl::io::JoinPath(testing::TempDir(), "oex_orchestration.pb");
  TF_CHECK_OK(tsl::WriteBinaryProto(tsl::Env::Default(), path, pipelines));
  EXPECT_THAT(GetOutputNames(path, "p1"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Empty name in named signature")));
}

TEST(GetManifestTest, SuccessWithValidManifestInTopDir) {
  const auto model_dir = tsl::io::JoinPath(testing::TempDir(), "model");
  ASSERT_OK(tsl::Env::Default()->CreateDir(model_dir));
  const auto version_file_path =
      tsl::io::JoinPath(model_dir, kOrbaxModelManifestVersionTxt);
  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), version_file_path,
                                   R"(
    version: "0.0.1"
    mime_type: "application/protobuf; type=orbax_model_manifest.Manifest"
    manifest_file_path: "manifest.pb")"));

  orbax_model_manifest::Manifest manifest;
  manifest.mutable_objects()->insert({"test_object", {}});

  ASSERT_OK(tsl::WriteBinaryProto(tsl::Env::Default(),
                                  tsl::io::JoinPath(model_dir, "manifest.pb"),
                                  manifest));

  TF_ASSERT_OK_AND_ASSIGN(orbax_model_manifest::Manifest got_manifest,
                          GetManifest(model_dir));
  EXPECT_THAT(got_manifest, EqualsProto(manifest));
}

TEST(GetManifestTest, SuccessWithValidManifestInSubDir) {
  const auto model_dir = tsl::io::JoinPath(testing::TempDir(), "model1");
  ASSERT_OK(tsl::Env::Default()->CreateDir(model_dir));
  const auto manifest_dir = tsl::io::JoinPath(model_dir, "sub/dir");
  ASSERT_OK(tsl::Env::Default()->RecursivelyCreateDir(manifest_dir));
  const auto version_file_path =
      tsl::io::JoinPath(model_dir, kOrbaxModelManifestVersionTxt);
  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), version_file_path,
                                   R"(
    version: "0.0.1"
    mime_type: "application/protobuf; type=orbax_model_manifest.Manifest"
    manifest_file_path: "sub/dir/manifest.pb")"));

  orbax_model_manifest::Manifest manifest;
  manifest.mutable_objects()->insert({"test_object", {}});

  const auto manifest_abs_path = tsl::io::JoinPath(manifest_dir, "manifest.pb");
  ASSERT_OK(
      tsl::WriteBinaryProto(tsl::Env::Default(), manifest_abs_path, manifest));

  TF_ASSERT_OK_AND_ASSIGN(orbax_model_manifest::Manifest got_manifest,
                          GetManifest(model_dir));
  EXPECT_THAT(got_manifest, EqualsProto(manifest));
}

TEST(GetManifestTest, VersionFileDoesNotExist) {
  const auto model_dir = tsl::io::JoinPath(testing::TempDir(), "model2");
  ASSERT_OK(tsl::Env::Default()->CreateDir(model_dir));
  EXPECT_THAT(GetManifest(model_dir), StatusIs(absl::StatusCode::kNotFound));
}

TEST(GetManifestTest, VersionFileMalformed) {
  const auto model_dir = tsl::io::JoinPath(testing::TempDir(), "model3");
  ASSERT_OK(tsl::Env::Default()->CreateDir(model_dir));
  const auto version_file_path =
      tsl::io::JoinPath(model_dir, kOrbaxModelManifestVersionTxt);

  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), version_file_path,
                                   R"(
    version "0.0.1"
    mime_type: "application/protobuf; type=orbax_model_manifest.Manifest"
    manifest_file_path: "sub/dir/manifest.pb")"));
  EXPECT_THAT(
      GetManifest(model_dir),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("Invalid line")));

  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), version_file_path,
                                   R"(
    version: "0.0.1"
    mime_type: "application/protobuf; type=orbax_model_manifest.Manifest"
    manifest_file_path: "sub/dir/manifest.pb"
    extra_key: "extra value")"));
  EXPECT_THAT(
      GetManifest(model_dir),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("Unknown key")));

  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), version_file_path,
                                   R"(
    version: '0.0.1'
    mime_type: "application/protobuf; type=orbax_model_manifest.Manifest"
    manifest_file_path: "sub/dir/manifest.pb")"));
  EXPECT_THAT(GetManifest(model_dir),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Value is not double-quoted")));

  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), version_file_path,
                                   R"(
    version: "0.0.2"
    mime_type: "application/protobuf; type=orbax_model_manifest.Manifest"
    manifest_file_path: "sub/dir/manifest.pb")"));
  EXPECT_THAT(GetManifest(model_dir),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unsupported version")));

  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), version_file_path,
                                   R"(
    version: "0.0.1"
    mime_type: "application/protobuf"
    manifest_file_path: "sub/dir/manifest.pb")"));
  EXPECT_THAT(GetManifest(model_dir),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unsupported type")));
}

TEST(GetManifestTest, ManifestPathMissing) {
  const auto model_dir = tsl::io::JoinPath(testing::TempDir(), "model4");
  ASSERT_OK(tsl::Env::Default()->CreateDir(model_dir));
  const auto version_file_path =
      tsl::io::JoinPath(model_dir, kOrbaxModelManifestVersionTxt);
  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), version_file_path,
                                   R"(
    version: "0.0.1"
    mime_type: "application/protobuf; type=orbax_model_manifest.Manifest"
    manifest_file_path: "")"));
  EXPECT_THAT(GetManifest(model_dir),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Manifest file path is empty")));
}

TEST(GetManifestTest, ManifestFileDoesNotExist) {
  const auto model_dir = tsl::io::JoinPath(testing::TempDir(), "model5");
  ASSERT_OK(tsl::Env::Default()->CreateDir(model_dir));
  const auto version_file_path =
      tsl::io::JoinPath(model_dir, kOrbaxModelManifestVersionTxt);
  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), version_file_path,
                                   R"(
    version: "0.0.1"
    mime_type: "application/protobuf; type=orbax_model_manifest.Manifest"
    manifest_file_path: "missing.pb")"));
  EXPECT_THAT(GetManifest(model_dir), StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace jsv
