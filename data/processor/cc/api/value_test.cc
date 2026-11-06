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

#include "data/processor/cc/api/value.h"  // from @xla_serving

#include <ostream>
#include <utility>

#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "data/processor/cc/api/type_info.h"  // from @xla_serving

namespace xla_serving::data {
namespace composite {
namespace api {
namespace {

TEST(ValueTest, SmallCopyable) {
  struct SmallCopyable {
    int v;
  };

  Value value(SmallCopyable{100});
  EXPECT_EQ(value.Get<SmallCopyable>().v, 100);

  Value value_copy(value);
  EXPECT_EQ(value_copy.Get<SmallCopyable>().v, 100);
  EXPECT_EQ(value.Get<SmallCopyable>().v, 100);

  Value value_move = std::move(value);
  EXPECT_EQ(value_move.Get<SmallCopyable>().v, 100);
  EXPECT_FALSE(value.HasValue());  // NOLINT

  ASSERT_TRUE(value_move.HasValue());
  value_move.Destroy<SmallCopyable>();
  EXPECT_FALSE(value_move.HasValue());

  value_move = SmallCopyable{100};
  EXPECT_EQ(value_move.Get<SmallCopyable>().v, 100);
}

TEST(ValueTest, HostBufferSizedInPlace) {
  struct HostBufferSized {
    alignas(8) char data[112];
  };
  static_assert(Value::IsInPlace<HostBufferSized>());
}

TEST(ValueTest, LargeCopyable) {
  constexpr char kData[] =
      "<<This line contains 32 bytes>>\n"
      "<<This line contains 32 bytes>>\n"
      "<<This line contains 32 bytes>>\n"
      "<<This line contains 32 bytes>>\n"
      "<<This line contains 32 bytes>>\n"
      "<<This line contains 32 bytes>>\n"
      "<<This line contains 32 bytes>>\n"
      "<<This line contains 32 bytes>>";

  static_assert(sizeof(kData) == 256);

  struct LargeCopyable {
    char data[256] =
        "<<This line contains 32 bytes>>\n"
        "<<This line contains 32 bytes>>\n"
        "<<This line contains 32 bytes>>\n"
        "<<This line contains 32 bytes>>\n"
        "<<This line contains 32 bytes>>\n"
        "<<This line contains 32 bytes>>\n"
        "<<This line contains 32 bytes>>\n"
        "<<This line contains 32 bytes>>";
  };
  static_assert(!Value::IsInPlace<LargeCopyable>());

  Value value(LargeCopyable{});
  EXPECT_EQ(absl::string_view(value.Get<LargeCopyable>().data), kData);

  Value value_copy = value;
  EXPECT_EQ(absl::string_view(value_copy.Get<LargeCopyable>().data), kData);
  EXPECT_EQ(absl::string_view(value.Get<LargeCopyable>().data), kData);

  Value value_move = std::move(value);
  EXPECT_EQ(absl::string_view(value_move.Get<LargeCopyable>().data), kData);
  EXPECT_FALSE(value.HasValue());  // NOLINT

  ASSERT_TRUE(value_move.HasValue());
  value_move.Destroy<LargeCopyable>();
  EXPECT_FALSE(value_move.HasValue());

  value_move = LargeCopyable{};
  EXPECT_EQ(absl::string_view(value_move.Get<LargeCopyable>().data), kData);
}

TEST(ValueTest, SmallMoveOnly) {
  struct SmallMoveOnly {
    int v;

    explicit SmallMoveOnly(int v) : v(v) {}
    SmallMoveOnly(const SmallMoveOnly&) = delete;
    SmallMoveOnly& operator=(const SmallMoveOnly&) = delete;
    SmallMoveOnly(SmallMoveOnly&&) = default;
    SmallMoveOnly& operator=(SmallMoveOnly&&) = default;
  };

  Value value(SmallMoveOnly(100));
  EXPECT_EQ(value.Get<SmallMoveOnly>().v, 100);

  Value value_move = std::move(value);
  EXPECT_EQ(value_move.Get<SmallMoveOnly>().v, 100);
  EXPECT_FALSE(value.HasValue());  // NOLINT
}

TEST(ValueTest, LargeMoveOnly) {
  constexpr char kData[] =
      "<<This line contains 32 bytes>>\n"
      "<<This line contains 32 bytes>>\n"
      "<<This line contains 32 bytes>>\n"
      "<<This line contains 32 bytes>>\n"
      "<<This line contains 32 bytes>>\n"
      "<<This line contains 32 bytes>>\n"
      "<<This line contains 32 bytes>>\n"
      "<<This line contains 32 bytes>>";

  static_assert(sizeof(kData) == 256);

  struct LargeMoveOnly {
    char data[256] =
        "<<This line contains 32 bytes>>\n"
        "<<This line contains 32 bytes>>\n"
        "<<This line contains 32 bytes>>\n"
        "<<This line contains 32 bytes>>\n"
        "<<This line contains 32 bytes>>\n"
        "<<This line contains 32 bytes>>\n"
        "<<This line contains 32 bytes>>\n"
        "<<This line contains 32 bytes>>";

    LargeMoveOnly() = default;
    LargeMoveOnly(const LargeMoveOnly&) = delete;
    LargeMoveOnly& operator=(const LargeMoveOnly&) = delete;
    LargeMoveOnly(LargeMoveOnly&&) = default;
    LargeMoveOnly& operator=(LargeMoveOnly&&) = default;
  };
  static_assert(!Value::IsInPlace<LargeMoveOnly>());

  Value value(LargeMoveOnly{});
  EXPECT_EQ(absl::string_view(value.Get<LargeMoveOnly>().data), kData);

  Value value_move = std::move(value);
  EXPECT_EQ(absl::string_view(value_move.Get<LargeMoveOnly>().data), kData);
  EXPECT_FALSE(value.HasValue());  // NOLINT
}

TEST(ValueTest, Error) {
  Value arg(100);

  arg.HandleError(arg);

  EXPECT_EQ(arg.Get<int>(), 100);

  struct Small {
    int* v = nullptr;

    void HandleError(Value* arg) { *v = arg->Get<int>(); }
  };

  int v = 0;

  Value value(Small{&v});

  value.HandleError(arg);

  EXPECT_EQ(v, 100);
  EXPECT_EQ(*value.Get<Small>().v, 100);

  struct Large {
    int* v = nullptr;
    char data[128];

    void HandleError(Value* arg) { *v = arg->Get<int>(); }
  };

  v = 0;

  value = Value(Large{&v});

  value.HandleError(arg);

  EXPECT_EQ(v, 100);
  EXPECT_EQ(*value.Get<Large>().v, 100);
}

TEST(ValueTest, TypeInfo) {
  Value empty;
  EXPECT_FALSE(empty.IsType<int>());
  EXPECT_FALSE(empty.IsType<float>());

  Value value(100);
  EXPECT_EQ(value.GetTypeInfo(), CreateTypeInfo<int>());
  EXPECT_TRUE(value.IsType<int>());
  EXPECT_TRUE(value.GetTypeInfo().IsType<int>());
  EXPECT_FALSE(value.IsType<float>());
  EXPECT_FALSE(value.GetTypeInfo().IsType<float>());

  Value value_copy = value;
  EXPECT_EQ(value_copy.GetTypeInfo(), CreateTypeInfo<int>());
  EXPECT_TRUE(value_copy.IsType<int>());
  EXPECT_TRUE(value_copy.GetTypeInfo().IsType<int>());
  EXPECT_FALSE(value_copy.IsType<float>());
  EXPECT_FALSE(value_copy.GetTypeInfo().IsType<float>());

  Value value_move = std::move(value);
  EXPECT_EQ(value_move.GetTypeInfo(), CreateTypeInfo<int>());
  EXPECT_TRUE(value_move.IsType<int>());
  EXPECT_TRUE(value_move.GetTypeInfo().IsType<int>());
  EXPECT_FALSE(value_move.IsType<float>());
  EXPECT_FALSE(value_move.GetTypeInfo().IsType<float>());
}

struct CustomStringifiable {
  int x;
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const CustomStringifiable& val) {
    sink.Append(absl::StrCat("Custom(", val.x, ")"));
  }
};

struct CustomOstreamable {
  int y;
  friend std::ostream& operator<<(std::ostream& os,
                                  const CustomOstreamable& val) {
    return os << "Ostream(" << val.y << ")";
  }
};

struct CustomNone {};

TEST(ValueTest, DebugStringAndStringify) {
  Value empty;
  EXPECT_EQ(empty.DebugString(), "(empty Value)");
  EXPECT_EQ(absl::StrCat(empty), "(empty Value)");

  Value v1(42);
  EXPECT_EQ(v1.DebugString(), "Value(42)");
  EXPECT_EQ(absl::StrCat(v1), "Value(42)");

  Value v2(CustomStringifiable{123});
  EXPECT_EQ(v2.DebugString(), "Value(Custom(123))");
  EXPECT_EQ(absl::StrCat(v2), "Value(Custom(123))");

  Value v3(CustomOstreamable{456});
  EXPECT_EQ(v3.DebugString(), "Value(Ostream(456))");
  EXPECT_EQ(absl::StrCat(v3), "Value(Ostream(456))");

  Value v4(CustomNone{});
  EXPECT_EQ(v4.DebugString(),
            absl::StrCat("Value(<Object of type ",
                         CreateTypeInfo<CustomNone>().DebugTypeName(), ">)"));
}

}  // namespace
}  // namespace api
}  // namespace composite
}  // namespace xla_serving::data
