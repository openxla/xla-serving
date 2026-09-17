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

#ifndef XLA_SERVING_DATA_PROCESSOR_CC_API_VALUE_H_
#define XLA_SERVING_DATA_PROCESSOR_CC_API_VALUE_H_

#include <new>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "data/processor/cc/api/type_info.h"  // from @xla_serving

namespace xla_serving::data {
namespace composite {
namespace api {

class Value;

namespace value_internal {

struct InPlaceStorageT {
  // Many tensor implementations like tensorflow::Tensor requires multiple
  // words, and we'd like to keep these values inplace.
  //
  // TODO(chky): Consider a better size for inplace storage.
  alignas(8) char data[56];
};

template <typename T>
using IsInPlaceStorage =
    std::integral_constant<bool, sizeof(T) <= sizeof(InPlaceStorageT) &&
                                     alignof(T) <= alignof(InPlaceStorageT) &&
                                     std::is_move_constructible_v<T>>;

// Since we type-erase the value to be put in class Value, we need to an enum
// value to select the operation that should be applied on the type-erased
// value.
enum class Action {
  kDestroy = 0,  // Destructor
  kCopy,         // Copy constructor/assignment
  kMove,         // Move constructor/assignment
  kError,        // Error handler
  kTypeInfo,     // Get type info
  kDebugString   // Get debug string
};

using HandlerFuncPtr = TypeInfo (*)(Action, Value*, Value*);

template <typename T>
class InPlaceHandler;
template <typename T>
class OutOfPlaceHandler;

template <typename T>
using Handler = std::conditional_t<IsInPlaceStorage<T>::value,
                                   InPlaceHandler<T>, OutOfPlaceHandler<T>>;

template <typename T, typename Enable = void>
struct HasHandleError : std::false_type {};

template <typename T>
struct HasHandleError<
    T, std::void_t<decltype(std::declval<T>().HandleError(nullptr))>>
    : std::true_type {};

}  // namespace value_internal

// A container for type-erased value. The value should be at least copy
// constructable to be put into this container. This container has both move and
// copy semantics, but if the concrete value does not support copy, calling the
// copy operations on this class will result in undefined behavior.
class alignas(64) Value {
 public:
  // Value is default constructible. The payload is unset in the default
  // constructed Value.
  Value() = default;

  Value(const Value&);
  Value& operator=(const Value&);
  Value(Value&&) noexcept;
  Value& operator=(Value&&) noexcept;

  // Construct Value and store `t` as the payload.
  template <typename T,
            std::enable_if_t<!std::is_same_v<std::decay_t<T>, Value>, int> = 0>
  explicit Value(T&& t);

  template <typename T,
            std::enable_if_t<!std::is_same_v<std::decay_t<T>, Value>, int> = 0>
  Value& operator=(T&& value) {
    Set(std::forward<T>(value));
    return *this;
  }

  ~Value();

  // Get() function returns the payload of the Value object in the requested
  // type.
  //
  // Dynamic type checking is performed in the debug mode.
  template <typename T>
  T& Get();

  template <typename T>
  const T& Get() const;

  // Emplace() constructs the payload object of type T in place with the given
  // args. If the value is already initialized, the original value will be
  // destroyed.
  template <typename T, typename... Args>
  void Emplace(Args&&... args);

  // Construct() constructs the payload object of type T in place with the given
  // args. The value should be uninitialized before calling this method.
  // Otherwise the behavior is undefined.
  template <typename T, typename... Args>
  void Construct(Args&&... args);

  // Destroy() destroys the payload object of type T. The value must be already
  // initialized with a value of type T. Otherwise the behavior is undefined.
  template <typename T>
  void Destroy();

  // Set() stores the argument `t` as the payload of Value.
  template <typename T>
  void Set(T&& t);

  // Reset the Value object to empty.
  void Reset();

  // Call T::HandleError() method on the underlying value of type T. If T does
  // not have a HandleError() method, this method does nothing.
  void HandleError(Value& arg);

  // Check if Value contains a payload.
  bool HasValue() const { return handler_ != nullptr; }

  // Check if Value contains object of type T.
  template <typename T>
  bool IsType() const;

  // Gets the type info of the payload.
  TypeInfo GetTypeInfo() const;

  // Gets the debug string representation of the payload.
  std::string DebugString() const;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Value& val) {
    sink.Append(val.DebugString());
  }

  friend std::ostream& operator<<(std::ostream& os, const Value& val) {
    return os << val.DebugString();
  }

  // Check if object of type T is stored in place.
  template <typename T>
  static constexpr bool IsInPlace() {
    return value_internal::IsInPlaceStorage<T>::value;
  }

 private:
  union {
    value_internal::InPlaceStorageT storage_{};
    void* value_;
  };
  value_internal::HandlerFuncPtr handler_ = nullptr;

  template <typename>
  friend class value_internal::InPlaceHandler;
  template <typename>
  friend class value_internal::OutOfPlaceHandler;
};

// We only optimize the code for 64-bit architectures for now.
static_assert(sizeof(Value) == 64 || sizeof(void*) != 8);

// -----------------------------------------------------------
// Implementation details.

namespace value_internal {

template <typename T>
TypeInfo GetTypeInfo();

template <typename T, typename = void>
struct IsAbslStringifiable : std::false_type {};

template <typename T>
struct IsAbslStringifiable<
    T, std::void_t<decltype(absl::StrCat(std::declval<const T&>()))>>
    : std::true_type {};

template <typename T, typename = void>
struct IsOstreamable : std::false_type {};

template <typename T>
struct IsOstreamable<T, std::void_t<decltype(std::declval<std::ostream&>()
                                             << std::declval<const T&>())>>
    : std::true_type {};

template <typename T>
std::string DebugStringPayload(const T& val) {
  if constexpr (IsAbslStringifiable<T>::value) {
    return absl::StrCat(val);
  } else if constexpr (IsOstreamable<T>::value) {
    std::ostringstream oss;
    oss << val;
    return oss.str();
  } else {
    return absl::StrCat("<Object of type ", GetTypeInfo<T>().DebugTypeName(),
                        ">");
  }
}

template <typename T,
          typename std::enable_if_t<HasHandleError<T>::value, int> = 0>
void HandleErrorInternal(Value* self, Value* arg) {
  std::move(self->Get<T>()).HandleError(arg);
}

template <typename T,
          typename std::enable_if_t<!HasHandleError<T>::value, int> = 0>
static void HandleErrorInternal(Value* self, Value* arg) {}

template <class T>
struct InPlaceHandler {
  template <typename... Args>
  static void Construct(Value* self, Args&&... args) {
    new (&self->storage_) T(std::forward<Args>(args)...);
    self->handler_ = &Handle;
  }

  static TypeInfo Handle(Action action, Value* self, Value* other) {
    switch (action) {
      case Action::kDestroy:
        Destroy(self);
        return TypeInfo();
      case Action::kCopy:
        Copy(self, other);
        return TypeInfo();
      case Action::kMove:
        Move(self, other);
        return TypeInfo();
      case Action::kError:
        HandleError(self, other);
        return TypeInfo();
      case Action::kTypeInfo:
        return GetTypeInfo<T>();
      case Action::kDebugString:
        other->Set(DebugStringPayload<T>(
            *std::launder(reinterpret_cast<const T*>(&self->storage_))));
        return TypeInfo();
    }
    return TypeInfo();
  }

  static void Destroy(Value* self) {
    DCHECK(self->HasValue());
    auto* p = std::launder(reinterpret_cast<T*>(&self->storage_));
    p->~T();
    self->handler_ = nullptr;
  }

  template <typename V,
            typename std::enable_if_t<std::is_copy_constructible_v<V>, int> = 0>
  static void CopyInternal(Value* self, Value* dest) {
    DCHECK(self->HasValue() && !dest->HasValue());
    Construct(dest, *std::launder(reinterpret_cast<const V*>(&self->storage_)));
  }

  template <typename V, typename std::enable_if_t<
                            !std::is_copy_constructible_v<V>, int> = 0>
  static void CopyInternal(Value* self, Value* dest) {
    LOG(FATAL)  // Crash Ok
        << "Copying a `xla_serving::Value` whose underlying type is not copyable is a "
           "runtime error.";
  }

  static void Copy(Value* self, Value* dest) { CopyInternal<T>(self, dest); }

  static void Move(Value* self, Value* dest) {
    DCHECK(self->HasValue() && !dest->HasValue());
    Construct(dest,
              std::move(*std::launder(reinterpret_cast<T*>(&self->storage_))));
    Destroy(self);
  }

  static void HandleError(Value* self, Value* arg) {
    HandleErrorInternal<T>(self, arg);
  }
};

template <class T>
struct OutOfPlaceHandler {
  template <typename... Args>
  static void Construct(Value* self, Args&&... args) {
    self->value_ = new T(std::forward<Args>(args)...);
    self->handler_ = &Handle;
  }

  static TypeInfo Handle(Action action, Value* self, Value* other) {
    switch (action) {
      case Action::kDestroy:
        Destroy(self);
        return TypeInfo();
      case Action::kCopy:
        Copy(self, other);
        return TypeInfo();
      case Action::kMove:
        Move(self, other);
        return TypeInfo();
      case Action::kError:
        HandleError(self, other);
        return TypeInfo();
      case Action::kTypeInfo:
        return GetTypeInfo<T>();
      case Action::kDebugString:
        other->Set(DebugStringPayload<T>(*static_cast<const T*>(self->value_)));
        return TypeInfo();
    }
    return TypeInfo();
  }

  static void Destroy(Value* self) {
    DCHECK(self->HasValue());
    delete static_cast<T*>(self->value_);
    self->handler_ = nullptr;
  }

  template <typename V,
            typename std::enable_if_t<std::is_copy_constructible_v<V>, int> = 0>
  static void CopyInternal(Value* self, Value* dest) {
    DCHECK(self->HasValue() && !dest->HasValue());
    Construct(dest, *static_cast<const V*>(self->value_));
  }

  template <typename V, typename std::enable_if_t<
                            !std::is_copy_constructible_v<V>, int> = 0>
  static void CopyInternal(Value* self, Value* dest) {
    LOG(FATAL)  // Crash Ok
        << "Copying a `xla_serving::Value` whose underlying type is not copyable is a "
           "runtime error.";
  }

  static void Copy(Value* self, Value* dest) { CopyInternal<T>(self, dest); }

  static void Move(Value* self, Value* dest) {
    DCHECK(self->HasValue() && !dest->HasValue());
    dest->value_ = self->value_;
    dest->handler_ = &Handle;
    self->handler_ = nullptr;
  }

  static void HandleError(Value* self, Value* arg) {
    HandleErrorInternal<T>(self, arg);
  }
};

template <typename T>
TypeInfo GetTypeInfo() {
  return CreateTypeInfo<T>();
}

}  // namespace value_internal

template <typename T,
          std::enable_if_t<!std::is_same_v<std::decay_t<T>, Value>, int>>
Value::Value(T&& t) {
  Construct<std::decay_t<T>>(std::forward<T>(t));
}

inline Value::Value(const Value& v) {
  if (v.HasValue())
    v.handler_(value_internal::Action::kCopy, const_cast<Value*>(&v), this);
}

inline Value& Value::operator=(const Value& v) {
  Reset();
  if (v.HasValue())
    v.handler_(value_internal::Action::kCopy, const_cast<Value*>(&v), this);
  return *this;
}

inline Value::Value(Value&& v) noexcept {
  if (v.HasValue()) v.handler_(value_internal::Action::kMove, &v, this);
}

inline Value& Value::operator=(Value&& v) noexcept {
  Reset();
  if (v.HasValue()) v.handler_(value_internal::Action::kMove, &v, this);
  return *this;
}

inline void Value::HandleError(Value& arg) {
  if (HasValue()) handler_(value_internal::Action::kError, this, &arg);
}

inline Value::~Value() { Reset(); }

template <typename T>
T& Value::Get() {
  return const_cast<T&>(static_cast<const Value*>(this)->Get<T>());
}

template <typename T>
const T& Value::Get() const {
  DCHECK(IsType<T>()) << "Expected type: " << typeid(T).name()
                      << ", but got type: " << GetTypeInfo().DebugTypeName();

  if constexpr (IsInPlace<T>()) {
    return *std::launder(reinterpret_cast<const T*>(&storage_));
  }

  return *static_cast<const T*>(value_);
}

// Emplace() constructs the payload object of type T in place with the given
// args.
template <typename T, typename... Args>
void Value::Emplace(Args&&... args) {
  Reset();
  Construct<std::decay_t<T>>(std::forward<Args>(args)...);
}

// Set() stores the argument `t` as the payload of Value.
template <typename T>
void Value::Set(T&& t) {
  Emplace<T>(std::forward<T>(t));
}

template <typename T, typename... Args>
void Value::Construct(Args&&... args) {
  DCHECK(!HasValue());
  static_assert(!std::is_same_v<T, Value>);
  value_internal::Handler<T>::Construct(this, std::forward<Args>(args)...);
}

template <typename T>
void Value::Destroy() {
  DCHECK(HasValue());
  DCHECK(IsType<T>());
  static_assert(!std::is_same_v<T, Value>);
  value_internal::Handler<T>::Destroy(this);
}

// Reset the Value object to empty.
inline void Value::Reset() {
  if (handler_ == nullptr) return;
  handler_(value_internal::Action::kDestroy, this, nullptr);
}

template <typename T>
bool Value::IsType() const {
  if (handler_ == nullptr) return false;
  return handler_(value_internal::Action::kTypeInfo, const_cast<Value*>(this),
                  nullptr)
      .IsType<T>();
}

}  // namespace api
}  // namespace composite
}  // namespace xla_serving::data

#endif  // XLA_SERVING_DATA_PROCESSOR_CC_API_VALUE_H_
