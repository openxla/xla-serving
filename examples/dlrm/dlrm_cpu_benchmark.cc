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

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "base/init_google.h"
#include "testing/base/public/benchmark.h"
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/flags/flag.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/random/distributions.h"  // from @abseil-cpp
#include "absl/random/random.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "runtime/core/default_allocator.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune
#include "runtime/core/value.h"  // from @neptune
#include "runtime/interpreter/interpreter.h"  // from @neptune
#include "runtime/tensorflow/tf_compatible_interpreter_create_options.h"  // from @neptune
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/ifrt/test_util.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "tensorflow/core/framework/tensor.h"  // from @org_tensorflow
#include "tensorflow/core/framework/tensor_shape.h"  // from @org_tensorflow

namespace {

std::string GetDefaultModelDir() {
  const char* home = getenv("HOME");
  if (home != nullptr) {
    return absl::StrCat(home, "/dlrm_benchmarks");
  }
  return "/tmp/dlrm_benchmarks";
}

}  // namespace

ABSL_FLAG(std::string, model_dir, GetDefaultModelDir(),
          "Directory where the models are exported.");
ABSL_FLAG(
    std::string, input_distribution, "uniform",
    "Distribution to use for sparse input ID generation: 'uniform' or 'zipf'.");

namespace {

xla::ifrt::Client* g_ifrt_client = nullptr;
tsl::thread::ThreadPool* g_worker_pool = nullptr;
tsl::thread::ThreadPool* g_comp_pool = nullptr;

struct ModelKey {
  std::string model_type;  // "baseline" or "jce"
  int embedding_dim;

  bool operator==(const ModelKey& o) const {
    return model_type == o.model_type && embedding_dim == o.embedding_dim;
  }

  template <typename H>
  friend H AbslHashValue(H h, const ModelKey& k) {
    return H::combine(std::move(h), k.model_type, k.embedding_dim);
  }
};

absl::Mutex g_model_cache_mu;
absl::flat_hash_map<ModelKey, std::shared_ptr<neptune::Interpreter>> g_model_cache;

std::shared_ptr<neptune::Interpreter> GetModel(const std::string& model_type,
                                           int embedding_dim,
                                           xla::ifrt::Client* ifrt_client,
                                           tsl::thread::ThreadPool* worker_pool,
                                           tsl::thread::ThreadPool* comp_pool) {
  absl::MutexLock lock(g_model_cache_mu);
  ModelKey key{model_type, embedding_dim};
  auto it = g_model_cache.find(key);
  if (it != g_model_cache.end()) {
    return it->second;
  }

  std::string model_path = absl::StrCat(absl::GetFlag(FLAGS_model_dir), "/",
                                        model_type, "/dim_", embedding_dim);
  LOG(INFO) << "Loading model from: " << model_path;

  auto create_options =
      std::make_unique<neptune::TfCompatibleInterpreterCreateOptions>();
  create_options->model_path = model_path;
  create_options->ifrt_client = std::shared_ptr<xla::ifrt::Client>(
      ifrt_client, [](xla::ifrt::Client*) {});
  create_options->host_ifrt_client = create_options->ifrt_client;
  create_options->execute_worker_thread_pool = worker_pool;
  create_options->computation_thread_pool = comp_pool;
  create_options->platform_name = "cpu";

  auto status_or_interpreter =
      neptune::Interpreter::Create(std::move(create_options));
  QCHECK_OK(status_or_interpreter.status())
      << "Failed to create interpreter for: " << model_path;

  auto interpreter =
      std::shared_ptr<neptune::Interpreter>(std::move(*status_or_interpreter));
  g_model_cache[key] = interpreter;
  return interpreter;
}

void BM_DLRM_Inference(benchmark::State& state, const std::string& model_type) {
  int batch_size = state.range(0);
  int embedding_dim = state.range(1);

  // Load model (cached)
  auto interpreter = GetModel(model_type, embedding_dim, g_ifrt_client,
                              g_worker_pool, g_comp_pool);

  // Generate synthetic inputs
  std::vector<int> valencies = {3, 2, 1, 2, 6, 1, 1,  1,   1,  7,  3, 8, 1,
                                6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1};
  std::vector<int> vocab_sizes = {
      400000, 39060,   17295,  7424,   20265,  3,     7122, 1543, 63,
      400000, 3067956, 405282, 10,     2209,   11938, 155,  4,    976,
      14,     400000,  400000, 400000, 590152, 12973, 108,  36};

  absl::flat_hash_map<std::string, neptune::Value> keyword_inputs;

  if (model_type == "baseline") {
    tensorflow::Tensor dense_features(
        tensorflow::DT_FLOAT, tensorflow::TensorShape({batch_size, 13}));
    auto dense_flat = dense_features.flat<float>();
    for (int i = 0; i < dense_flat.size(); ++i) {
      dense_flat(i) = 0.5f;
    }
    keyword_inputs["dense_features"] = neptune::Value(dense_features);

    absl::BitGen bitgen;
    std::string dist_type = absl::GetFlag(FLAGS_input_distribution);
    for (int i = 0; i < 26; ++i) {
      int valency = valencies[i];
      int vocab_size = vocab_sizes[i];
      tensorflow::Tensor sparse_feat(
          tensorflow::DT_INT32, tensorflow::TensorShape({batch_size, valency}));
      auto sparse_flat = sparse_feat.flat<int32_t>();
      for (int j = 0; j < sparse_flat.size(); ++j) {
        if (dist_type == "zipf") {
          sparse_flat(j) =
              absl::Zipf<int32_t>(bitgen, vocab_size - 1, 1.1, 1.0);
        } else {
          sparse_flat(j) = absl::uniform_int_distribution<int32_t>(
              0, vocab_size - 1)(bitgen);
        }
      }
      keyword_inputs[std::to_string(i)] = neptune::Value(sparse_feat);
    }
  } else {
    neptune::HostBuffer dense_features(
        xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
        xla::ifrt::Shape({batch_size, 13}),
        neptune::DefaultAllocator::GetInstance());
    auto dense_flat = dense_features.flat<float>();
    for (int i = 0; i < dense_flat.size(); ++i) {
      dense_flat(i) = 0.5f;
    }
    keyword_inputs["dense_features"] = neptune::Value(dense_features);

    absl::BitGen bitgen;
    std::string dist_type = absl::GetFlag(FLAGS_input_distribution);
    for (int i = 0; i < 26; ++i) {
      int valency = valencies[i];
      int vocab_size = vocab_sizes[i];
      neptune::HostBuffer sparse_feat(
          xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
          xla::ifrt::Shape({batch_size, valency}),
          neptune::DefaultAllocator::GetInstance());
      auto sparse_flat = sparse_feat.flat<int32_t>();
      for (int j = 0; j < sparse_flat.size(); ++j) {
        if (dist_type == "zipf") {
          sparse_flat(j) =
              absl::Zipf<int32_t>(bitgen, vocab_size - 1, 1.1, 1.0);
        } else {
          sparse_flat(j) = absl::uniform_int_distribution<int32_t>(
              0, vocab_size - 1)(bitgen);
        }
      }
      keyword_inputs[std::to_string(i)] = neptune::Value(sparse_feat);
    }
  }

  // Warmup run to compile StableHLO graph
  auto warmup_status =
      interpreter->Run("serving_default",
                       /*positional_inputs=*/{}, keyword_inputs);
  QCHECK_OK(warmup_status.status())
      << "Warmup run failed for " << model_type << " batch_size=" << batch_size
      << " dim=" << embedding_dim;

  // Loop of measurements
  for (auto s : state) {
    auto outputs = interpreter->Run("serving_default",
                                    /*positional_inputs=*/{}, keyword_inputs);
    benchmark::DoNotOptimize(outputs);
  }

  state.SetItemsProcessed(state.iterations() * batch_size);
}

void BM_Baseline(benchmark::State& state) {
  BM_DLRM_Inference(state, "baseline");
}

void BM_JCE(benchmark::State& state) { BM_DLRM_Inference(state, "jce"); }

void RegisterBenchmarks() {
  std::vector<int> batch_sizes = {1, 64, 512, 4096};
  std::vector<int> embedding_dims = {16, 64, 128};

  for (int bs : batch_sizes) {
    for (int dim : embedding_dims) {
      // Register Baseline benchmark
      auto* b_baseline =
          benchmark::RegisterBenchmark(
              absl::StrCat("BM_Baseline/BS_", bs, "/Dim_", dim).c_str(),
              BM_Baseline)
              ->Args({bs, dim});
      for (int threads : {1, 2, 4, 8, 12, 16, 20, 24}) {
        b_baseline->Threads(threads);
      }

      // Register JCE benchmark
      auto* b_jce =
          benchmark::RegisterBenchmark(
              absl::StrCat("BM_JCE/BS_", bs, "/Dim_", dim).c_str(), BM_JCE)
              ->Args({bs, dim});
      for (int threads : {1, 2, 4, 8, 12, 16, 20, 24}) {
        b_jce->Threads(threads);
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  InitGoogle(argv[0], &argc, &argv, true);

  std::shared_ptr<xla::ifrt::Client> client_keep_alive;
  {
    auto client_or = xla::ifrt::test_util::GetClient();
    QCHECK_OK(client_or.status());
    g_ifrt_client = client_or->get();
    client_keep_alive = std::move(*client_or);
  }

  g_worker_pool = new tsl::thread::ThreadPool(
      tsl::Env::Default(), tsl::ThreadOptions(), "ExecutionThreadPool", 16);
  g_comp_pool = new tsl::thread::ThreadPool(
      tsl::Env::Default(), tsl::ThreadOptions(), "ComputationThreadPool", 16);

  RegisterBenchmarks();
  RunSpecifiedBenchmarks();

  // Clear cached model interpreters while thread pools and clients are still
  // alive
  {
    absl::MutexLock lock(g_model_cache_mu);
    g_model_cache.clear();
  }

  delete g_worker_pool;
  delete g_comp_pool;
  g_worker_pool = nullptr;
  g_comp_pool = nullptr;

  g_ifrt_client = nullptr;
  client_keep_alive.reset();

  return 0;
}
