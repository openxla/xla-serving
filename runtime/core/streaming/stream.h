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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_CORE_STREAMING_STREAM_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_CORE_STREAMING_STREAM_H_

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <queue>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/functional/any_invocable.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_format.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/time/time.h"  // from @abseil-cpp
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "runtime/core/host_buffer.h"  // from @neptune
#include "xla/tsl/platform/threadpool_interface.h"  // from @xla

namespace neptune {

struct StreamedResult {
  absl::flat_hash_map<std::string, neptune::HostBuffer> host_buffers;
  absl::Time enqueued_time;
};

class StreamControllerInterface {
 public:
  explicit StreamControllerInterface(std::string controller_address)
      : controller_address_(std::move(controller_address)) {}
  virtual ~StreamControllerInterface() = default;

  absl::string_view controller_address() const { return controller_address_; }

  virtual void RecordDequeueLatency(absl::string_view model_name,
                                    absl::Duration latency) {}

  virtual void RecordCallbackLatency(absl::string_view model_name,
                                     absl::Duration latency) {}

 private:
  std::string controller_address_;
};

class StreamWorkerInterface {
 public:
  explicit StreamWorkerInterface(std::string controller_address)
      : controller_address_(std::move(controller_address)) {}
  virtual ~StreamWorkerInterface() = default;

  absl::string_view controller_address() const { return controller_address_; }

  virtual void RecordSendLatency(absl::string_view model_name,
                                 absl::Duration latency) {}
  virtual absl::Status InvokeStreamCallback(
      int64_t callback_id, const std::vector<std::string>& names,
      const std::vector<std::pair<int64_t, std::vector<neptune::HostBuffer>>>&
          responses) = 0;

 private:
  std::string controller_address_;
};

class ScopedStreamCallback;

class StreamInterfaceFactory {
 public:
  using CreateWorkerStreamInterfaceFn =
      std::function<absl::StatusOr<std::unique_ptr<StreamWorkerInterface>>(
          absl::string_view)>;

  void RegisterController(
      absl::AnyInvocable<
          absl::StatusOr<std::unique_ptr<StreamControllerInterface>>() const>
          interface_factory) {
    absl::MutexLock lock(mu_);
    controller_interface_factory_ = std::move(interface_factory);
  }

  absl::StatusOr<std::unique_ptr<StreamControllerInterface>>
  CreateControllerStreamInterface() const {
    absl::MutexLock lock(mu_);
    return controller_interface_factory_();
  }

  void RegisterWorker(CreateWorkerStreamInterfaceFn interface_factory) {
    absl::MutexLock lock(mu_);
    worker_interface_factory_ = std::move(interface_factory);
  }

  CreateWorkerStreamInterfaceFn CreateWorkerStreamInterface() const {
    absl::MutexLock lock(mu_);
    return worker_interface_factory_;
  }

 private:
  mutable absl::Mutex mu_;
  absl::AnyInvocable<
      absl::StatusOr<std::unique_ptr<StreamControllerInterface>>() const>
      controller_interface_factory_ ABSL_GUARDED_BY(mu_) = []() {
        return absl::UnimplementedError(
            "The factory for StreamControllerInterface is not registered.");
      };

  CreateWorkerStreamInterfaceFn worker_interface_factory_ ABSL_GUARDED_BY(mu_) =
      [](absl::string_view) {
        return absl::UnimplementedError(
            "The factory for StreamWorkerInterface is not registered.");
      };
};

StreamInterfaceFactory& GetGlobalStreamInterfaceFactory();

class StreamCallbackRegistry {
 public:
  explicit StreamCallbackRegistry(
      std::unique_ptr<StreamControllerInterface> interface)
      : interface_(std::move(interface)) {
    DCHECK(interface_);
  }

  absl::StatusOr<ScopedStreamCallback> Register(
      absl::string_view model_name, int64_t callback_id, int64_t step_id,
      absl::AnyInvocable<
          void(absl::flat_hash_map<std::string, neptune::HostBuffer>)>
          callback);

  absl::Status Invoke(tsl::thread::ThreadPoolInterface* thread_pool,
                      int64_t callback_id, int64_t step_id,
                      StreamedResult result);

  StreamControllerInterface& stream_interface() const { return *interface_; }

 private:
  friend class ScopedStreamCallback;

  class CallbackState {
   public:
    CallbackState(StreamCallbackRegistry* registry,
                  absl::string_view model_name, int64_t callback_id,
                  int64_t step_id,
                  absl::AnyInvocable<
                      void(absl::flat_hash_map<std::string, neptune::HostBuffer>)>
                      callback)
        : registry_(registry),
          model_name_(model_name),
          callback_id_(callback_id),
          step_id_(step_id),
          callback_(std::move(callback)) {
      DCHECK(registry_);
    }

    absl::Status Invoke(tsl::thread::ThreadPoolInterface* thread_pool,
                        StreamedResult result);

    void Close();

   private:
    StreamControllerInterface& interface() {
      return registry_->stream_interface();
    }
    void InvokeCallback(StreamedResult result);
    void ProcessQueue();

    StreamCallbackRegistry* registry_ = nullptr;
    std::string model_name_;
    int64_t callback_id_;
    int64_t step_id_;
    absl::AnyInvocable<void(absl::flat_hash_map<std::string, neptune::HostBuffer>)>
        callback_;

    absl::Mutex mu_;
    bool closed_ ABSL_GUARDED_BY(mu_) = false;
    int num_outstanding_ ABSL_GUARDED_BY(mu_) = 0;
    bool is_processing_ ABSL_GUARDED_BY(mu_) = false;
    std::queue<StreamedResult> pending_results_ ABSL_GUARDED_BY(mu_);
  };

  std::unique_ptr<CallbackState> Unregister(int64_t callback_id,
                                            int64_t step_id);

  std::unique_ptr<StreamControllerInterface> interface_;

  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::pair<int64_t, int64_t>,
                      std::unique_ptr<CallbackState>>
      stream_callbacks_ ABSL_GUARDED_BY(mu_);
};

absl::StatusOr<StreamCallbackRegistry*> GetGlobalStreamCallbackRegistry();

// Sets the callback ID in the MLIR module's streaming custom calls.
// Returns the callback_id if the module contains streaming custom calls,
// or std::nullopt if the module has no streaming custom calls (in which case
// the module is not mutated and streaming is disabled for this module).
absl::StatusOr<std::optional<int64_t>> SetStreamCallbackId(
    mlir::ModuleOp module, int64_t callback_id);

class ScopedStreamCallback {
 public:
  ScopedStreamCallback() = default;

  ScopedStreamCallback(ScopedStreamCallback&& other);
  ScopedStreamCallback& operator=(ScopedStreamCallback&& other);

  ~ScopedStreamCallback() { Unregister(); }

 private:
  friend class StreamCallbackRegistry;

  explicit ScopedStreamCallback(StreamCallbackRegistry* registry,
                                int64_t callback_id, int64_t step_id)
      : registry_(registry), callback_id_(callback_id), step_id_(step_id) {}

  void Unregister();

  StreamCallbackRegistry* registry_ = nullptr;
  std::optional<int64_t> callback_id_;
  int64_t step_id_ = -1;
};

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_CORE_STREAMING_STREAM_H_
