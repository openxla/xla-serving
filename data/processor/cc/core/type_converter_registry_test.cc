#include "data/processor/cc/core/type_converter_registry.h"  // from @neptune

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "data/processor/cc/api/type_converter.h"  // from @neptune
#include "data/processor/cc/api/type_info.h"  // from @neptune
#include "data/test_utils/status_macros.h"  // from @neptune

namespace neptune::data {
namespace composite {
namespace core {

class TypeConverterRegistryTest : public ::testing::Test {
 protected:
  TypeConverterRegistry registry_;
};

namespace {

struct TestTypeA {};
struct TestTypeB {};

class ConverterAB : public api::TypeConverter<TestTypeA, TestTypeB> {
 public:
  absl::StatusOr<TestTypeB> Convert(const TestTypeA& input) const override {
    return TestTypeB();
  }
};

TEST_F(TypeConverterRegistryTest, RegisterDuplicateRegistrationFails) {
  // Register first time (should succeed)
  ASSERT_TRUE(
      registry_.RegisterConverter(std::make_unique<ConverterAB>()).ok());

  // Register second time (should fail)
  auto register_duplicate =
      registry_.RegisterConverter(std::make_unique<ConverterAB>());
  EXPECT_FALSE(register_duplicate.ok());
  EXPECT_EQ(register_duplicate.code(), absl::StatusCode::kAlreadyExists);
}

TEST_F(TypeConverterRegistryTest, LookupSucceeds) {
  ASSERT_TRUE(
      registry_.RegisterConverter(std::make_unique<ConverterAB>()).ok());
  ASSERT_OK_AND_ASSIGN(auto converter, registry_.LookupConverter(
                                           api::CreateTypeInfo<TestTypeA>(),
                                           api::CreateTypeInfo<TestTypeB>()));
  EXPECT_NE(converter, nullptr);
}

TEST_F(TypeConverterRegistryTest, LookupFails) {
  ASSERT_TRUE(
      registry_.RegisterConverter(std::make_unique<ConverterAB>()).ok());
  auto converter = registry_.LookupConverter(api::CreateTypeInfo<TestTypeB>(),
                                             api::CreateTypeInfo<TestTypeA>());
  EXPECT_FALSE(converter.ok());
  EXPECT_EQ(converter.status().code(), absl::StatusCode::kNotFound);
}

}  // namespace

}  // namespace core
}  // namespace composite
}  // namespace neptune::data
