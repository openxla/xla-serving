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

#include "runtime/core/streaming/stream.h"  // from @neptune

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/functional/any_invocable.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_format.h"  // from @abseil-cpp
#include "absl/strings/str_split.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/time/clock.h"  // from @abseil-cpp
#include "absl/time/time.h"  // from @abseil-cpp
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"  // from @llvm-project
#include "runtime/core/host_buffer.h"  // from @neptune
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "xla/executable_run_options.h"  // from @xla
#include "xla/ffi/ffi.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "tsl/platform/random.h"  // from @tsl
#include "tsl/profiler/lib/traceme.h"  // from @tsl

namespace ffi = xla::ffi;

namespace neptune {
namespace {

constexpr int64_t kInvalidStepId = -1;
constexpr absl::string_view kNeptuneStreamOpName = "neptune_stream";
constexpr absl::string_view kBackendConfigAttr = "mhlo.backend_config";
constexpr absl::string_view kControllerAddressAttr = "controller_address";
constexpr absl::string_view kCallbackIdAttr = "callback_id";
constexpr absl::string_view kOutputNamesAttr = "output_names";

}  // namespace

absl::StatusOr<std::optional<int64_t>> CreateStreamCallbackId(
    mlir::ModuleOp module) {
  mlir::Builder builder(module.getContext());

  std::vector<mlir::stablehlo::CustomCallOp> ops;
  module->walk([&](mlir::stablehlo::CustomCallOp op) {
    if (op.getCallTargetName().str() != kNeptuneStreamOpName) {
      return;
    }
    ops.push_back(op);
  });

  if (ops.empty()) {
    return std::nullopt;
  }

  TF_ASSIGN_OR_RETURN(auto* stream_callback_registry,
                      GetGlobalStreamCallbackRegistry());
  auto& stream_interface = stream_callback_registry->stream_interface();

  auto controller_address = stream_interface.controller_address();
  auto controller_address_attr = builder.getStringAttr(controller_address);

  const int64_t callback_id = static_cast<int64_t>(tsl::random::New64());
  auto callback_id_attr = builder.getI64IntegerAttr(callback_id);

  for (auto op : ops) {
    mlir::DictionaryAttr backend_config =
        op->getAttrOfType<mlir::DictionaryAttr>(kBackendConfigAttr);

    llvm::SmallVector<mlir::NamedAttribute, 4> new_attrs;
    if (backend_config) {
      new_attrs.append(backend_config.getValue().begin(),
                       backend_config.getValue().end());
    }

    new_attrs.push_back({builder.getStringAttr(kControllerAddressAttr),
                         controller_address_attr});
    new_attrs.push_back(
        {builder.getStringAttr(kCallbackIdAttr), callback_id_attr});

    op->setAttr(kBackendConfigAttr, builder.getDictionaryAttr(new_attrs));
  }

  return callback_id;
}

absl::Status StreamCallbackRegistry::CallbackState::Invoke(
    tsl::thread::ThreadPool* thread_pool, StreamedResult result) {
  {
    absl::MutexLock lock(mu_);
    if (closed_) {
      return absl::InternalError(
          "Failed to invole the callback that is closed.");
    }
    ++num_outstanding_;
  }
  thread_pool->Schedule([this, result = std::move(result)]() mutable {
    InvokeCallback(std::move(result));

    absl::MutexLock lock(mu_);
    --num_outstanding_;
  });
  return absl::OkStatus();
}

void StreamCallbackRegistry::CallbackState::Close() {
  {
    absl::MutexLock lock(mu_);
    closed_ = true;

    auto not_running = [this]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
      return num_outstanding_ == 0;
    };
    mu_.Await(absl::Condition(&not_running));
  }
}

void StreamCallbackRegistry::CallbackState::InvokeCallback(
    StreamedResult result) {
  absl::Duration dequeue_latency = absl::Now() - result.enqueued_time;
  interface().RecordDequeueLatency(model_name_, dequeue_latency);

  tsl::profiler::TraceMe trace_me("StreamCallbackInvocation");
  trace_me.AppendMetadata([&]() {
    return tsl::profiler::TraceMeEncode({
        {"callback_id", callback_id_},
        {"step_id", step_id_},
    });
  });

  absl::Time start_time = absl::Now();
  callback_(std::move(result.host_buffers));
  interface().RecordCallbackLatency(model_name_, absl::Now() - start_time);
}

absl::StatusOr<ScopedStreamCallback> StreamCallbackRegistry::Register(
    absl::string_view model_name, int64_t callback_id, int64_t step_id,
    absl::AnyInvocable<void(absl::flat_hash_map<std::string, neptune::HostBuffer>)>
        callback) {
  absl::MutexLock l(mu_);

  const auto [it, inserted] =
      stream_callbacks_.insert({std::make_pair(callback_id, step_id), nullptr});
  if (!inserted) {
    return absl::AlreadyExistsError(absl::StrCat(
        "Stream callback ", callback_id, " @ ", step_id, " already exists"));
  }

  it->second = std::make_unique<CallbackState>(this, model_name, callback_id,
                                               step_id, std::move(callback));
  return ScopedStreamCallback(this, callback_id, step_id);
}

absl::Status StreamCallbackRegistry::Invoke(
    tsl::thread::ThreadPool* thread_pool, int64_t callback_id, int64_t step_id,
    StreamedResult result) {
  absl::MutexLock lock(mu_);
  auto iter = stream_callbacks_.find({callback_id, step_id});
  if (iter == stream_callbacks_.end()) {
    return absl::NotFoundError(absl::StrCat(
        "Stream callback ", callback_id, " @ ", step_id,
        " does not exist; this usually indicates that a streaming signature "
        "was called by a non-streaming request"));
  }

  auto* state = iter->second.get();
  DCHECK(state);
  return state->Invoke(thread_pool, std::move(result));
}

std::unique_ptr<StreamCallbackRegistry::CallbackState>
StreamCallbackRegistry::Unregister(int64_t callback_id, int64_t step_id) {
  absl::MutexLock l(mu_);
  const auto it = stream_callbacks_.find({callback_id, step_id});
  if (it == stream_callbacks_.end()) {
    return nullptr;
  }
  auto state = std::move(it->second);
  stream_callbacks_.erase(it);
  return state;
}

ScopedStreamCallback::ScopedStreamCallback(ScopedStreamCallback&& other)
    : registry_(other.registry_),
      callback_id_(other.callback_id_),
      step_id_(other.step_id_) {
  other.callback_id_ = std::nullopt;
  other.step_id_ = kInvalidStepId;
}

ScopedStreamCallback& ScopedStreamCallback::operator=(
    ScopedStreamCallback&& other) {
  Unregister();

  registry_ = other.registry_;
  callback_id_ = other.callback_id_;
  step_id_ = other.step_id_;
  other.callback_id_ = std::nullopt;
  other.step_id_ = kInvalidStepId;

  return *this;
}

void ScopedStreamCallback::Unregister() {
  if (!callback_id_.has_value()) {
    return;
  }

  tsl::profiler::TraceMe trace_me("ScopedStreamCallback::Unregister");
  trace_me.AppendMetadata([&]() {
    return tsl::profiler::TraceMeEncode({
        {"callback_id", *callback_id_},
        {"step_id", step_id_},
    });
  });

  DCHECK(registry_);
  auto state = registry_->Unregister(*callback_id_, step_id_);
  DCHECK(state);

  state->Close();

  callback_id_.reset();
}

StreamInterfaceFactory& GetGlobalStreamInterfaceFactory() {
  static auto* stream_interface_factory = new StreamInterfaceFactory;
  return *stream_interface_factory;
}

absl::StatusOr<StreamCallbackRegistry*> GetGlobalStreamCallbackRegistry() {
  static auto res = []() -> absl::StatusOr<StreamCallbackRegistry*> {
    TF_ASSIGN_OR_RETURN(
        auto interface,
        GetGlobalStreamInterfaceFactory().CreateControllerStreamInterface());
    return new StreamCallbackRegistry(std::move(interface));
  }();
  return res;
}

struct StreamState {
  std::string controller_address;
  std::unique_ptr<StreamWorkerInterface> stream;

  static absl::StatusOr<std::unique_ptr<StreamState>> Instantiate(
      ffi::Dictionary attrs) {
    TF_ASSIGN_OR_RETURN(auto controller_address,
                        attrs.get<absl::string_view>("controller_address"));

    auto state = std::make_unique<StreamState>();
    state->controller_address = std::string(controller_address);
    TF_ASSIGN_OR_RETURN(
        state->stream,
        GetGlobalStreamInterfaceFactory().CreateWorkerStreamInterface()(
            controller_address));
    return state;
  }
};

XLA_FFI_DEFINE_HANDLER(kInstantiateStream, StreamState::Instantiate,
                       ffi::Ffi::BindInstantiate().Attrs());

absl::Status Stream(xla::RunId run_id, ffi::Dictionary attrs,
                    ffi::RemainingArgs args, ffi::RemainingRets rets,
                    StreamState* state) {
  if (rets.size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Expected 1 output (token), got %d", rets.size()));
  }

  TF_ASSIGN_OR_RETURN(int64_t cb_id, attrs.get<int64_t>("callback_id"));

  auto output_names_attr = attrs.get<absl::string_view>(kOutputNamesAttr);
  std::vector<std::string> names;
  if (output_names_attr.ok()) {
    names = absl::StrSplit(*output_names_attr, ',');
  } else {
    for (size_t i = 0; i < args.size(); ++i) {
      names.push_back(absl::StrCat("output_", i));
    }
  }

  if (names.size() != args.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected %d names, got %d", args.size(), names.size()));
  }

  absl::flat_hash_map<std::string, neptune::HostBuffer> host_buffers;
  for (size_t i = 0; i < args.size(); ++i) {
    auto arg = args.get<ffi::AnyBuffer>(i);
    if (!arg.ok()) return arg.status();

    auto buffer = std::make_shared<std::vector<char>>(arg->size_bytes());
    std::memcpy(buffer->data(), arg->untyped_data(), arg->size_bytes());

    std::vector<int64_t> dims(arg->dimensions().begin(),
                              arg->dimensions().end());
    xla::ifrt::Shape shape(dims);
    TF_ASSIGN_OR_RETURN(xla::ifrt::DType dtype,
                        xla::ifrt::ToDType(arg->element_type()));

    neptune::HostBuffer host_buffer(buffer->data(), dtype, shape, buffer);

    host_buffers[names[i]] = std::move(host_buffer);
  }

  int64_t step_id = run_id.ToInt();

  std::vector<neptune::HostBuffer> buffers;
  buffers.reserve(names.size());
  for (const auto& name : names) {
    auto it = host_buffers.find(name);
    if (it == host_buffers.end()) {
      return absl::InternalError(
          absl::StrCat("Host buffer not found for name: ", name));
    }
    buffers.push_back(std::move(it->second));
  }

  std::vector<std::pair<int64_t, std::vector<neptune::HostBuffer>>> responses;
  responses.push_back({step_id, std::move(buffers)});

  return state->stream->InvokeStreamCallback(cb_id, names, responses);
}

XLA_FFI_DEFINE_HANDLER(kStream, Stream,
                       ffi::Ffi::Bind()
                           .Ctx<xla::RunId>()
                           .Attrs<ffi::Dictionary>()
                           .RemainingArgs()
                           .RemainingRets()
                           .Ctx<xla::ffi::State<StreamState>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), kNeptuneStreamOpName, "Host",
                         {
                             /*instantiate=*/kInstantiateStream,
                             /*prepare=*/nullptr,
                             /*initialize=*/nullptr,
                             /*execute=*/kStream,
                         });
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), kNeptuneStreamOpName, "CUDA",
                         {
                             /*instantiate=*/kInstantiateStream,
                             /*prepare=*/nullptr,
                             /*initialize=*/nullptr,
                             /*execute=*/kStream,
                         });

}  // namespace neptune
