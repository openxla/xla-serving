#ifndef THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_CC_API_TYPE_CONVERTER_H_
#define THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_CC_API_TYPE_CONVERTER_H_

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "data/processor/cc/api/type_info.h"  // from @neptune
#include "data/processor/cc/api/value.h"  // from @neptune

namespace neptune::data {
namespace composite {
namespace api {

// TypeConverterBase is the type-erased base class for all TypeConverters.
class TypeConverterBase {
 public:
  virtual ~TypeConverterBase() = default;

  virtual absl::StatusOr<Value> ConvertUntyped(const Value& input) const = 0;

  virtual TypeInfo InputType() const = 0;
  virtual TypeInfo OutputType() const = 0;
};

// TypeConverter converts a value of type InputT to OutputT.
// TypeConverters should be implemented as stateless.
template <typename InputT, typename OutputT>
class TypeConverter : public TypeConverterBase {
 public:
  // Converts a value of type InputT to OutputT.
  virtual absl::StatusOr<OutputT> Convert(const InputT& input) const = 0;

  absl::StatusOr<Value> ConvertUntyped(const Value& input) const override {
    // We intentionally trade the extra copy of OutputT into Value
    // for improved type-safety.
    auto converted = Convert(input.Get<InputT>());
    if (!converted.ok()) {
      return converted.status();
    }
    return Value(*std::move(converted));
  }

  TypeInfo InputType() const override { return CreateTypeInfo<InputT>(); }
  TypeInfo OutputType() const override { return CreateTypeInfo<OutputT>(); }
};

}  // namespace api
}  // namespace composite
}  // namespace neptune::data

#endif  // THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_CC_API_TYPE_CONVERTER_H_
