#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_OPTIONAL_FUNCTION_HANDLER_FACTORY_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_OPTIONAL_FUNCTION_HANDLER_FACTORY_H_

#include <stdbool.h>

#include <memory>
#include <string>

#include "runtime/interpreter/optional_function_handler.h"  // from @neptune
#include "absl/base/no_destructor.h"  // from @abseil-cpp
#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp

namespace neptune {

// A factory class interface to create FunctionHandler.
//
// The factory class is registered by `RegisterOptionalFunctionHandlerFactory`
// and the handler is retrieved by `GetFunctionHandler`.
// Factory classes can be registered with string keys. All implementations of
// `OptionalFunctionHandlerFactory` should be default constructible.
class OptionalFunctionHandlerFactory {
 public:
  virtual ~OptionalFunctionHandlerFactory() = default;
  virtual std::unique_ptr<OptionalFunctionHandler> GetHandler() = 0;
};

// A static singleton registry class for OptionalFunctionHandlerFactory.
// The registry is composed at link-time.
class OptionalFunctionHandlerFactoryRegistry {
 public:
  static OptionalFunctionHandlerFactoryRegistry& Instance();

  void Register(absl::string_view key, OptionalFunctionHandlerFactory* factory);

  absl::flat_hash_map<std::string, OptionalFunctionHandlerFactory*>
  GetFactories() const;

 private:
  friend class absl::NoDestructor<OptionalFunctionHandlerFactoryRegistry>;
  OptionalFunctionHandlerFactoryRegistry() = default;

  mutable absl::Mutex registry_mu_;
  absl::flat_hash_map<std::string, OptionalFunctionHandlerFactory*> factories_
      ABSL_GUARDED_BY(registry_mu_);
};

// Registers the factory class with the given key. This should be called at link
// time.
template <typename Factory>
void RegisterOptionalFunctionHandlerFactory(absl::string_view key) {
  OptionalFunctionHandlerFactoryRegistry::Instance().Register(key,
                                                              new Factory());
}

// A router class which maintains a mapping of processor keys to
// singleton OptionalFunctionHandler instances. Each OptionalFunctionHandler is
// created once and maintained in an in-memory map.
class OptionalFunctionHandlerRouter {
 public:
  OptionalFunctionHandlerRouter() {
    const absl::flat_hash_map<std::string, OptionalFunctionHandlerFactory*>
        factories =
            OptionalFunctionHandlerFactoryRegistry::Instance().GetFactories();

    for (const auto& [key, factory] : factories) {
      handlers_[key] = factory->GetHandler();
    }
  }

  // Returns the handler for the given key. Returns an internal error if the
  // handler is not found.
  absl::StatusOr<OptionalFunctionHandler*> GetHandler(absl::string_view key);

 private:
  absl::Mutex router_mu_;
  absl::flat_hash_map<std::string, std::unique_ptr<OptionalFunctionHandler>>
      handlers_ ABSL_GUARDED_BY(router_mu_);
};

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_OPTIONAL_FUNCTION_HANDLER_FACTORY_H_
