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

// Minimal client for a Gemma3 model exported by examples/gemma_keras/export.py.
// One Predict RPC runs the whole in-graph generation and returns token IDs.
//
// Sends token IDs, not text (no C++ tokenizer). --max_seq_len and
// --eos_token_id must match the export.
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @abseil-cpp
#include "absl/flags/parse.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "include/grpcpp/channel.h"  // from @grpc
#include "include/grpcpp/client_context.h"  // from @grpc
#include "include/grpcpp/create_channel.h"  // from @grpc
#include "include/grpcpp/security/credentials.h"  // from @grpc
#include "include/grpcpp/support/status.h"  // from @grpc
#include "protos/grpc/host_buffer.pb.h"  // from @xla_serving
#include "protos/grpc/model.pb.h"  // from @xla_serving
#include "protos/grpc/predict.pb.h"  // from @xla_serving
#include "protos/grpc/prediction_service.grpc.pb.h"  // from @xla_serving
#include "runtime/serving/gcp/auth/utils.h"  // from @xla_serving
#include "xla/python/ifrt/dtype.pb.h"  // from @xla
#include "xla/python/ifrt/shape.pb.h"  // from @xla

ABSL_FLAG(std::string, server_address, "", "Server IP address and port.");
ABSL_FLAG(std::string, model_name, "", "Name of the model to query.");
ABSL_FLAG(std::string, prompt_tokens_file, "",
          "Path to a raw int32 binary file containing prompt token IDs. When "
          "empty, the built-in kDefaultPrompt is used.");
ABSL_FLAG(int32_t, max_seq_len, 128,
          "Padded prompt length. Must match the value used at export time.");
ABSL_FLAG(int32_t, eos_token_id, 106,
          "Token ID that terminates decode early once every row has emitted "
          "it. Default is Gemma3's <end_of_turn> (106).");
ABSL_FLAG(std::string, server_cert_chain_path, "",
          "Path to the server's certificate chain file for SSL/TLS. If not "
          "provided, the client falls back on default trust stores.");
ABSL_FLAG(bool, use_insecure_grpc_credentials, false,
          "Whether to use insecure credentials for the server. WARNING: THIS "
          "SHOULD NOT BE USED IN PRODUCTION.");

namespace {

// "What is the capital of France?" in the Gemma3 instruct chat template:
//   <bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n
// For another prompt, tokenize the template (the tokenizer doesn't add <bos>)
// and write raw int32 for --prompt_tokens_file:
//   tok = keras_hub.models.Gemma3Tokenizer.from_preset("gemma3_instruct_4b")
//   ids = [tok.start_token_id] + list(tok(text_without_bos))
//   np.asarray(ids, dtype=np.int32).tofile("prompt.bin")
// Decode the output with tok.detokenize(ids).
constexpr int32_t kDefaultPrompt[] = {
    2,                                         // <bos>
    105,  2364, 107,                           // <start_of_turn>user\n
    3689, 563,  506, 5279, 529, 7001, 236881,  // What is the capital of France?
    106,  107,                                 // <end_of_turn>\n
    105,  4368, 107,                           // <start_of_turn>model\n
};

absl::StatusOr<std::vector<int32_t>> LoadPromptTokens(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Could not open prompt tokens file: ", path));
  }
  // tellg() can be -1 or bogus (e.g. a directory); the checks below catch it.
  const std::streamoff file_size = file.tellg();
  if (file_size <= 0 || file_size % sizeof(int32_t) != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Prompt tokens file size (", file_size,
        ") must be a non-zero multiple of sizeof(int32_t)."));
  }
  file.seekg(0);
  std::vector<int32_t> tokens(file_size / sizeof(int32_t));
  file.read(reinterpret_cast<char*>(tokens.data()), file_size);
  if (!file) {
    return absl::InternalError(
        absl::StrCat("Short read from prompt tokens file: ", path));
  }
  return tokens;
}

void AddInput(xla_serving::serving::PredictRequest* request,
              absl::string_view name, const std::vector<int64_t>& dims,
              std::string raw_data) {
  xla_serving::HostBufferProto buffer;
  buffer.mutable_dtype()->set_kind(xla::ifrt::DTypeProto::KIND_S32);
  auto* shape = buffer.mutable_shape();
  for (int64_t d : dims) shape->add_dims(d);
  buffer.set_raw_data(std::move(raw_data));
  (*request->mutable_inputs())[std::string(name)] = std::move(buffer);
}

// Encodes an int32 as a [1]-shaped buffer (the batcher needs a batch dim).
std::string Int32Bytes(int32_t v) {
  std::string buf(sizeof(int32_t), '\0');
  std::memcpy(buf.data(), &v, sizeof(int32_t));
  return buf;
}

}  // namespace

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  const std::string server_address = absl::GetFlag(FLAGS_server_address);
  const std::string model_name = absl::GetFlag(FLAGS_model_name);
  const int32_t max_seq_len = absl::GetFlag(FLAGS_max_seq_len);
  const int32_t eos_token_id = absl::GetFlag(FLAGS_eos_token_id);

  if (server_address.empty()) {
    LOG(ERROR) << "--server_address must be provided.";
    return 1;
  }
  if (model_name.empty()) {
    LOG(ERROR) << "--model_name must be provided.";
    return 1;
  }

  const std::string prompt_tokens_file =
      absl::GetFlag(FLAGS_prompt_tokens_file);
  std::vector<int32_t> prompt_tokens;
  if (!prompt_tokens_file.empty()) {
    auto tokens_or = LoadPromptTokens(prompt_tokens_file);
    if (!tokens_or.ok()) {
      LOG(ERROR) << tokens_or.status();
      return 1;
    }
    prompt_tokens = *std::move(tokens_or);
  } else {
    prompt_tokens.assign(std::begin(kDefaultPrompt), std::end(kDefaultPrompt));
  }
  const int32_t prompt_len = static_cast<int32_t>(prompt_tokens.size());
  if (prompt_len < 1 || prompt_len > max_seq_len) {
    LOG(ERROR) << "Invalid prompt_len=" << prompt_len
               << "; must satisfy 1 <= prompt_len <= max_seq_len ("
               << max_seq_len << ").";
    return 1;
  }

  auto creds_or = xla_serving::serving::CreateChannelCredentials(
      absl::GetFlag(FLAGS_server_cert_chain_path),
      absl::GetFlag(FLAGS_use_insecure_grpc_credentials));
  if (!creds_or.ok()) {
    LOG(ERROR) << "Failed to create channel credentials: " << creds_or.status();
    return 1;
  }
  std::shared_ptr<grpc::Channel> channel =
      grpc::CreateChannel(server_address, *creds_or);
  std::unique_ptr<xla_serving::serving::PredictionService::Stub> stub =
      xla_serving::serving::PredictionService::NewStub(channel);
  LOG(INFO) << "Connected to " << server_address;

  // token_ids is the prompt right-padded with zeros to [1, max_seq_len].
  xla_serving::serving::PredictRequest request;
  request.mutable_model_spec()->set_name(model_name);
  std::string token_buf(
      static_cast<size_t>(max_seq_len) * sizeof(int32_t), '\0');
  std::memcpy(token_buf.data(), prompt_tokens.data(),
              prompt_len * sizeof(int32_t));
  AddInput(&request, "token_ids", {1, max_seq_len}, std::move(token_buf));
  AddInput(&request, "prompt_len", {1}, Int32Bytes(prompt_len));
  AddInput(&request, "eos_token_id", {1}, Int32Bytes(eos_token_id));

  xla_serving::serving::PredictResponse response;
  grpc::ClientContext ctx;
  grpc::Status s = stub->Predict(&ctx, request, &response);
  if (!s.ok()) {
    LOG(ERROR) << "Predict RPC failed: " << s.error_code() << ": "
               << s.error_message();
    return 1;
  }

  auto it = response.outputs().find("generated_ids");
  if (it == response.outputs().end()) {
    LOG(ERROR) << "'generated_ids' not found in response.";
    return 1;
  }
  // Batch is 1, so the whole [1, max_new_tokens] buffer is row 0.
  absl::string_view raw = it->second.raw_data();
  if (raw.size() % sizeof(int32_t) != 0) {
    LOG(ERROR) << "'generated_ids' raw_data size (" << raw.size()
               << ") is not a multiple of sizeof(int32_t).";
    return 1;
  }
  // Copy: protobuf strings aren't guaranteed 4-byte aligned.
  std::vector<int32_t> data(raw.size() / sizeof(int32_t));
  std::memcpy(data.data(), raw.data(), raw.size());
  std::vector<int32_t> tokens;
  for (size_t i = 0; i < data.size(); ++i) {
    if (data[i] == eos_token_id || (data[i] == 0 && i > 0)) break;  // EOS/pad
    tokens.push_back(data[i]);
  }

  LOG(INFO) << "Prompt length: " << prompt_len;
  LOG(INFO) << "Generated " << tokens.size() << " tokens: ["
            << absl::StrJoin(tokens, ", ") << "]";
  return 0;
}
