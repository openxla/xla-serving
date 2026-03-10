#include "runtime/interpreter/optional_function_handler_factory.h"  // from @neptune

#include <memory>
#include <string>

#include "runtime/interpreter/optional_function_handler.h"  // from @neptune
#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl

namespace jsv {

OptionalFunctionHandlerFactoryRegistry&
OptionalFunctionHandlerFactoryRegistry::Instance() {
  static absl::NoDestructor<OptionalFunctionHandlerFactoryRegistry> instance;
  return *instance;
}

void OptionalFunctionHandlerFactoryRegistry::Register(
    absl::string_view key, OptionalFunctionHandlerFactory* factory) {
  absl::MutexLock lock(registry_mu_);
  auto [_, inserted] = factories_.try_emplace(key, factory);
  CHECK(inserted) << "An OptionalFunctionHandlerFactory is already registered "
                     "for key: "
                  << key;
}

absl::flat_hash_map<std::string, OptionalFunctionHandlerFactory*>
OptionalFunctionHandlerFactoryRegistry::GetFactories() const {
  absl::MutexLock lock(registry_mu_);
  return factories_;
}

absl::StatusOr<OptionalFunctionHandler*>
OptionalFunctionHandlerRouter::GetHandler(absl::string_view key) {
  absl::MutexLock lock(router_mu_);
  if (auto it = handlers_.find(key); it != handlers_.end()) {
    return it->second.get();
  }
  return absl::InternalError(absl::StrCat(
      "No `OptionalFunctionHandler` is found for key: ", key,
      ". This can be because `InitializeHandlers` was not called or the "
      "handler is not registered."));
}

}  // namespace jsv
