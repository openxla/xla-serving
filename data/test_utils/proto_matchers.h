#ifndef THIRD_PARTY_NEPTUNE_DATA_TEST_UTILS_PROTO_MATCHERS_H_
#define THIRD_PARTY_NEPTUNE_DATA_TEST_UTILS_PROTO_MATCHERS_H_

#include <string>
#include <type_traits>

#include "gmock/gmock.h"
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"

namespace neptune::data {
namespace test_utils {

template <typename ProtoType, typename T>
bool CompareProtos(const ProtoType& actual, const T& expected,
                   bool ignore_repeated_field_ordering) {
  if constexpr (std::is_convertible_v<T, std::string> ||
                std::is_convertible_v<T, absl::string_view>) {
    ProtoType expected_proto;
    if (!google::protobuf::TextFormat::ParseFromString(expected, &expected_proto)) {
      return false;
    }
    google::protobuf::util::MessageDifferencer differencer;
    if (ignore_repeated_field_ordering) {
      differencer.set_repeated_field_comparison(
          google::protobuf::util::MessageDifferencer::AS_SET);
    }
    return differencer.Compare(actual, expected_proto);
  } else {
    google::protobuf::util::MessageDifferencer differencer;
    if (ignore_repeated_field_ordering) {
      differencer.set_repeated_field_comparison(
          google::protobuf::util::MessageDifferencer::AS_SET);
    }
    return differencer.Compare(actual, expected);
  }
}

MATCHER_P(EqualsProto, expected, "") {
  return CompareProtos(arg, expected, /*ignore_repeated_field_ordering=*/false);
}

MATCHER_P(EqualsProtoIgnoringRepeatedFieldOrdering, expected, "") {
  return CompareProtos(arg, expected, /*ignore_repeated_field_ordering=*/true);
}

}  // namespace test_utils
}  // namespace neptune::data

#endif  // THIRD_PARTY_NEPTUNE_DATA_TEST_UTILS_PROTO_MATCHERS_H_
