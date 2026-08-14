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

// Custom op and CPU kernel registration for RegexSplitWithOffsets.
//
// Branched directly from
// `//third_party/tensorflow_text/core/kernels/regex_split_kernels.cc`,
// `//third_party/tensorflow_text/core/kernels/regex_split.cc`, and
// `//third_party/tensorflow_text/core/ops/regex_split_ops.cc`.
//
// Why this file exists:
// SavedModel tokenizers (such as KerasHub / TensorFlow Text regex-based
// splitters) require the `RegexSplitWithOffsets` custom op to execute text
// preprocessing inside the Neptune C++ model server. In OSS builds, Neptune
// does not bundle or link the full tensorflow-text C++ repository due to
// external Bazel dependency constraints. This self-contained kernel provides
// the op directly in OSS Neptune serving binaries.
//
// In Google3 builds, this kernel is NOT directly linked into
// `init_lib_tensorflow` because Google3 server binaries (e.g. Orbax model
// servers) already link the canonical `//third_party/tensorflow_text:...`
// libraries; statically registering `RegexSplitWithOffsets` twice in the same
// binary causes a fatal OpRegistry duplicate registration error
// (ALREADY_EXISTS).
// Instead, `copy.bara.sky` links this `:regex_split_op` target into
// `init_lib_tensorflow` specifically during OSS export.
//
// TODO: Figure out how to link the required tensorflow-text
// kernels in OSS without copying or maintaining duplicate kernel
// implementations ourselves.

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "third_party/re2/re2.h"
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/macros.h"  // from @xla
#include "tensorflow/core/framework/op.h"  // from @org_tensorflow
#include "tensorflow/core/framework/op_kernel.h"  // from @org_tensorflow
#include "tensorflow/core/framework/op_requires.h"  // from @org_tensorflow
#include "tensorflow/core/framework/shape_inference.h"  // from @org_tensorflow
#include "tensorflow/core/framework/tensor.h"  // from @org_tensorflow
#include "tensorflow/core/framework/tensor_shape.h"  // from @org_tensorflow
#include "tensorflow/core/framework/tensor_types.h"  // from @org_tensorflow
#include "tensorflow/core/framework/types.h"  // from @org_tensorflow
#include "tensorflow/core/platform/errors.h"  // from @org_tensorflow
#include "tensorflow/core/platform/mutex.h"  // from @org_tensorflow
#include "tensorflow/core/platform/tstring.h"  // from @org_tensorflow
#include "tensorflow/core/platform/types.h"  // from @org_tensorflow
#include "tsl/platform/thread_annotations.h"  // from @tsl

namespace tensorflow {
namespace text {
namespace {

absl::Status RegexSplitOpShape(shape_inference::InferenceContext* c) {
  shape_inference::ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));

  for (int i = 0; i < c->num_outputs(); ++i) {
    c->set_output(i, c->UnknownShapeOfRank(1));
  }
  return absl::OkStatus();
}

template <typename T>
void RegexSplitImpl(absl::string_view input, const RE2& re2,
                    bool include_delimiter, const RE2& include_delim_regex,
                    std::vector<absl::string_view>* tokens,
                    std::vector<T>* begin_offsets,
                    std::vector<T>* end_offsets) {
  absl::string_view leftover = input;
  absl::string_view last_end = leftover;

  // Keep looking for split points until we have reached the end of the input.
  absl::string_view extracted_delim_token;
  while (RE2::FindAndConsume(&leftover, re2, &extracted_delim_token)) {
    absl::string_view token(last_end.data(),
                            extracted_delim_token.data() - last_end.data());
    bool has_non_empty_token = !token.empty();
    bool should_include_delim =
        include_delimiter && include_delim_regex.FullMatch(
                                 extracted_delim_token, include_delim_regex);
    last_end = leftover;

    // Mark the end of the previous token, only if there was something.
    if (has_non_empty_token) {
      tokens->push_back(token);
      // Mark the end of the last token
      begin_offsets->push_back(token.data() - input.data());
      end_offsets->push_back(token.data() + token.length() - input.data());
    }

    if (should_include_delim) {
      // If desired, include the deliminator as a token.
      tokens->push_back(extracted_delim_token);
      // Mark the end of the token at the end of the beginning of the delimiter.
      begin_offsets->push_back(extracted_delim_token.data() - input.data());
      end_offsets->push_back(extracted_delim_token.data() +
                             extracted_delim_token.length() - input.data());
    }
  }

  // Close the last token.
  if (!leftover.empty()) {
    tokens->push_back(leftover);
    begin_offsets->push_back(leftover.data() - input.data());
    end_offsets->push_back(leftover.data() + leftover.length() - input.data());
  }
}

void RegexSplit(absl::string_view input, const RE2& re2, bool include_delimiter,
                const RE2& include_delim_regex,
                std::vector<absl::string_view>* tokens,
                std::vector<int64>* begin_offsets,
                std::vector<int64>* end_offsets) {
  RegexSplitImpl(input, re2, include_delimiter, include_delim_regex, tokens,
                 begin_offsets, end_offsets);
}

}  // namespace

REGISTER_OP("RegexSplitWithOffsets")
    .Input("input: string")
    .Input("delim_regex_pattern: string")
    .Input("keep_delim_regex_pattern: string")
    .Output("tokens: string")
    .Output("begin_offsets: int64")
    .Output("end_offsets: int64")
    .Output("row_splits: int64")
    .SetShapeFn(RegexSplitOpShape)
    .Doc(R"doc(
Split strings using a regex as the delimiter.

See https://github.com/google/re2/wiki/Syntax for the full list of supported
expressions.
)doc");

class RegexSplitOp : public tensorflow::OpKernel {
 public:
  explicit RegexSplitOp(tensorflow::OpKernelConstruction* ctx)
      : tensorflow::OpKernel(ctx) {}

  void Compute(tensorflow::OpKernelContext* ctx) override {
    bool should_keep_delim;
    std::shared_ptr<RE2> delim_re;
    std::shared_ptr<RE2> keep_delim_re;

    // get regular expressions from input
    const Tensor* delim_regex_pattern_tensor;
    OP_REQUIRES_OK(
        ctx, ctx->input("delim_regex_pattern", &delim_regex_pattern_tensor));
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(delim_regex_pattern_tensor->shape()),
                errors::InvalidArgument(
                    "Pattern must be scalar, but received ",
                    delim_regex_pattern_tensor->shape().DebugString()));
    const string delim_regex_pattern =
        delim_regex_pattern_tensor->flat<tstring>()(0);
    delim_re = CachedDelimRE2(delim_regex_pattern);
    OP_REQUIRES(
        ctx, delim_re->ok(),
        errors::InvalidArgument("Invalid pattern: ", delim_regex_pattern,
                                ", error: ", delim_re->error()));

    const Tensor* keep_delim_regex_pattern_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("keep_delim_regex_pattern",
                                   &keep_delim_regex_pattern_tensor));
    OP_REQUIRES(
        ctx,
        TensorShapeUtils::IsScalar(keep_delim_regex_pattern_tensor->shape()),
        errors::InvalidArgument(
            "Pattern must be scalar, but received ",
            keep_delim_regex_pattern_tensor->shape().DebugString()));
    const string keep_delim_regex_pattern =
        keep_delim_regex_pattern_tensor->flat<tstring>()(0);
    keep_delim_re = CachedKeepDelimRE2(keep_delim_regex_pattern);
    OP_REQUIRES(
        ctx, keep_delim_re->ok(),
        errors::InvalidArgument("Invalid pattern: ", keep_delim_regex_pattern,
                                ", error: ", keep_delim_re->error()));

    should_keep_delim = keep_delim_re->pattern().empty() ? false : true;

    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    const auto& input_flat = input_tensor->flat<tstring>();

    std::vector<int64> begin_offsets;
    std::vector<int64> end_offsets;
    std::vector<absl::string_view> tokens;
    std::vector<int64> row_splits;
    row_splits.push_back(0);

    for (size_t i = 0; i < input_flat.size(); ++i) {
      RegexSplit(absl::string_view(input_flat(i).data()), *delim_re,
                 should_keep_delim, *keep_delim_re, &tokens, &begin_offsets,
                 &end_offsets);
      row_splits.push_back(begin_offsets.size());
    }

    // Emit the flat Tensors needed to construct RaggedTensors for tokens,
    // start, end offsets.
    std::vector<int64> tokens_shape;
    tokens_shape.push_back(tokens.size());

    std::vector<int64> offsets_shape;
    offsets_shape.push_back(begin_offsets.size());

    std::vector<int64> row_splits_shape;
    row_splits_shape.push_back(row_splits.size());

    Tensor* output_tokens_tensor = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("tokens", TensorShape(tokens_shape),
                                        &output_tokens_tensor));
    auto output_tokens = output_tokens_tensor->flat<tstring>();

    Tensor* output_begin_offsets_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("begin_offsets", TensorShape(offsets_shape),
                                  &output_begin_offsets_tensor));
    auto output_begin_offsets = output_begin_offsets_tensor->flat<int64>();

    Tensor* output_end_offsets_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("end_offsets", TensorShape(offsets_shape),
                                  &output_end_offsets_tensor));
    auto output_end_offsets = output_end_offsets_tensor->flat<int64>();

    Tensor* output_row_splits_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("row_splits", TensorShape(row_splits_shape),
                                  &output_row_splits_tensor));
    auto output_row_splits = output_row_splits_tensor->flat<int64>();

    // Copy outputs to Tensors.
    for (size_t i = 0; i < tokens.size(); ++i) {
      const auto& token = tokens[i];
      output_tokens(i) = tstring(token.data(), token.length());
    }

    for (size_t i = 0; i < begin_offsets.size(); ++i) {
      output_begin_offsets(i) = begin_offsets[i];
    }

    for (size_t i = 0; i < end_offsets.size(); ++i) {
      output_end_offsets(i) = end_offsets[i];
    }

    for (size_t i = 0; i < row_splits.size(); ++i) {
      output_row_splits(i) = row_splits[i];
    }
  }

 private:
  std::shared_ptr<RE2> CachedDelimRE2(const string& pattern) {
    {
      tf_shared_lock l(delim_mu_);
      if (delim_re_ != nullptr && delim_re_->pattern() == pattern) {
        return delim_re_;
      }
    }
    // Construct the new RE2 object before acquiring the lock.
    auto regex = std::make_shared<RE2>(pattern);
    {
      mutex_lock l(delim_mu_);
      // Swap instead of assigning so that we destruct the old
      // RE2 object (when necessary) after releasing the lock.
      delim_re_.swap(regex);
      return delim_re_;
    }
  }

  std::shared_ptr<RE2> CachedKeepDelimRE2(const string& pattern) {
    {
      tf_shared_lock l(keep_delim_mu_);
      if (keep_delim_re_ != nullptr && keep_delim_re_->pattern() == pattern) {
        return keep_delim_re_;
      }
    }
    // Construct the new RE2 object before acquiring the lock.
    auto regex = std::make_shared<RE2>(pattern);
    {
      mutex_lock l(keep_delim_mu_);
      // Swap instead of assigning so that we destruct the old
      // RE2 object (when necessary) after releasing the lock.
      keep_delim_re_.swap(regex);
      return keep_delim_re_;
    }
  }

  mutex delim_mu_;
  std::shared_ptr<RE2> delim_re_ TF_GUARDED_BY(delim_mu_);

  mutex keep_delim_mu_;
  std::shared_ptr<RE2> keep_delim_re_ TF_GUARDED_BY(keep_delim_mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(RegexSplitOp);
};

REGISTER_KERNEL_BUILDER(
    Name("RegexSplitWithOffsets").Device(tensorflow::DEVICE_CPU), RegexSplitOp);

}  // namespace text
}  // namespace tensorflow
