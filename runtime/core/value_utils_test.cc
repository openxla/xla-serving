#include "third_party/neptune/runtime/core/value_utils.h"

#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/neptune/runtime/core/value.h"
#include "third_party/neptune/runtime/core/value_matcher.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"

namespace jsv {
namespace {

using ::testing::ElementsAre;

// A move-only type to test the move-only type support.
class MoveOnly {
 public:
  explicit MoveOnly(int value) : value_(value) {}
  MoveOnly(const MoveOnly&) = delete;
  MoveOnly& operator=(const MoveOnly&) = delete;
  MoveOnly(MoveOnly&&) = default;
  MoveOnly& operator=(MoveOnly&&) = default;

  int value() const { return value_; }

 private:
  int value_;
};

TEST(PackAsValuesTest, Move) {
  std::vector<int> objects = {1, 2};
  std::vector<Value> values = internal::PackAsValues(std::move(objects));
  EXPECT_THAT(values, ElementsAre(ValueOfType<int>(1), ValueOfType<int>(2)));
}

TEST(PackAsValuesTest, MoveOnlyType) {
  std::vector<MoveOnly> objects;
  int num_objects = 10;
  for (int i = 0; i < num_objects; ++i) {
    objects.push_back(MoveOnly(i));
  }
  std::vector<Value> values = internal::PackAsValues(std::move(objects));
  ASSERT_EQ(values.size(), num_objects);
  for (int i = 0; i < num_objects; ++i) {
    EXPECT_TRUE(values[i].IsType<MoveOnly>());
    EXPECT_EQ(values[i].Get<MoveOnly>().value(), i);
  }
}

TEST(PackAsValuesTest, MapOfMoveOnlyType) {
  absl::flat_hash_map<int, MoveOnly> objects;
  int num_objects = 10;
  for (int i = 0; i < num_objects; ++i) {
    objects.emplace(i, MoveOnly(i));
  }
  absl::flat_hash_map<int, Value> values =
      internal::PackAsValues(std::move(objects));
  ASSERT_EQ(values.size(), num_objects);
  for (int i = 0; i < num_objects; ++i) {
    auto it = values.find(i);
    ASSERT_NE(it, values.end());
    EXPECT_TRUE(it->second.IsType<MoveOnly>());
    EXPECT_EQ(values[i].Get<MoveOnly>().value(), i);
  }
}

TEST(UnpackFromValuesTest, Move) {
  std::vector<Value> values = {Value(1), Value(2)};
  TF_ASSERT_OK_AND_ASSIGN(std::vector<int> objects,
                          internal::UnpackFromValues<int>(std::move(values)));
  EXPECT_THAT(objects, ElementsAre(1, 2));
}

TEST(UnpackFromValuesTest, Copy) {
  const std::vector<Value> values = {Value(1), Value(2)};
  TF_ASSERT_OK_AND_ASSIGN(std::vector<int> objects,
                          internal::UnpackFromValues<int>(values));
  EXPECT_THAT(objects, ElementsAre(1, 2));
}

TEST(UnpackFromValuesTest, MoveOnlyType) {
  std::vector<Value> values;
  int num_objects = 10;
  for (int i = 0; i < num_objects; ++i) {
    values.push_back(Value(MoveOnly(i)));
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<MoveOnly> objects,
      internal::UnpackFromValues<MoveOnly>(std::move(values)));
  for (int i = 0; i < num_objects; ++i) {
    EXPECT_EQ(objects[i].value(), i);
  }
}

}  // namespace
}  // namespace jsv
