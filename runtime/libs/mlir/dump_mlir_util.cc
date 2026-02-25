#include "third_party/neptune/runtime/libs/mlir/dump_mlir_util.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include "third_party/absl/log/log.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/match.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/synchronization/mutex.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/ADT/STLExtras.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/ADT/StringMap.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/ADT/StringRef.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/ADT/Twine.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/Support/FormatVariadic.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/Support/raw_ostream.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/Operation.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/OperationSupport.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Pass/Pass.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Pass/PassManager.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Support/LLVM.h"
#include "third_party/tensorflow/compiler/xla/tsl/lib/io/buffered_file.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/env.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/file_system.h"
#include "third_party/tensorflow/tsl/platform/path.h"

using llvm::raw_ostream;

namespace jsv {
namespace {

struct NameCounts {
  absl::Mutex counts_mutex;
  llvm::StringMap<int64_t> counts;
};

std::string MakeUniqueFilename(std::string name) {
  static NameCounts& instance = *new NameCounts;

  // Remove illegal characters from `name`.
  for (int i = 0, e = name.size(); i < e; ++i) {
    char ch = name[i];
    if (ch == '/' || ch == '[' || ch == ']' || ch == '*' || ch == '?' ||
        ch == '\\') {
      name[i] = '_';
    }
  }

  int count;
  {
    absl::MutexLock lock(instance.counts_mutex);
    count = instance.counts[name]++;
  }

  std::string filename = name;
  if (count > 0) {
    filename = llvm::formatv("{0}_{1}", filename, count).str();
  }
  filename = llvm::Twine(filename).concat(".mlir").str();
  return filename;
}

// Simple raw_ostream that prints to stderr.
struct LogInfoRawStream : public llvm::raw_ostream {
  LogInfoRawStream() { SetUnbuffered(); }
  ~LogInfoRawStream() override = default;
  uint64_t current_pos() const override { return 0; }

  void write_impl(const char* ptr, size_t size) override {
    fprintf(stderr, "%.*s", static_cast<int>(size), ptr);
  }
};

// Simple raw_ostream that prints to a file.
struct WritableFileRawStream : public llvm::raw_ostream {
  explicit WritableFileRawStream(std::unique_ptr<tsl::WritableFile> file)
      : file(std::move(file)) {
    SetUnbuffered();
  }
  ~WritableFileRawStream() override = default;

  uint64_t current_pos() const override {
    int64_t position;
    if (file->Tell(&position).ok()) {
      return position;
    } else {
      // MLIR uses os.tell() to determine whether something was written by
      // a subroutine or not, so it's important we have a working current_pos().
      LOG(WARNING)
          << "Couldn't query file position. Stream might be malformed.\n";
      return -1;
    }
  }

  void write_impl(const char* ptr, size_t size) override {
    // Write the file if it is still valid. If the write fails, null out the
    // file to avoid encountering another error.
    if (file && !file->Append(absl::string_view(ptr, size)).ok()) {
      file = nullptr;
    }
  }

  // The file being written to.
  std::unique_ptr<tsl::WritableFile> file;
};

}  // namespace

absl::Status CreateFileForDumping(llvm::StringRef name,
                                  std::unique_ptr<raw_ostream>* os,
                                  std::string* filepath,
                                  llvm::StringRef dirname) {
  std::string dir;
  if (!dirname.empty())
    dir = std::string(dirname);
  else
    dir = GetDumpDirFromEnvVar();

  if (dir.empty()) {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        "(JSV_DUMP_GRAPH_PREFIX not specified)");
  }

  if (dir == kCrashReproducerStdErr) {
    *os = std::make_unique<LogInfoRawStream>();
    *filepath =
        llvm::formatv("(stderr; requested filename: '{0}')", name).str();
    return absl::Status();
  }

  // Get a valid file path to dump with.
  tsl::Env* env = tsl::Env::Default();
  absl::Status status = env->RecursivelyCreateDir(dir);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to create '" << dir
                 << "' directory for dumping: " << status;
    return absl::Status(absl::StatusCode::kUnavailable, "(unavailable)");
  }
  *filepath = tsl::io::JoinPath(dir, MakeUniqueFilename(std::string(name)));

  // Try to open the file and generate a raw_ostream.
  std::unique_ptr<tsl::WritableFile> file;
  status = env->NewWritableFile(*filepath, &file);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to create file '" << filepath << "': " << status;
    return absl::Status(absl::StatusCode::kUnavailable, "(unavailable)");
  }
  file = std::make_unique<tsl::BufferedWritableFile>(std::move(file));
  *os = std::make_unique<WritableFileRawStream>(std::move(file));
  return absl::Status();
}

// Prints the pass pipeline of `pass_manager` to `os`.
void PrintPassPipeline(const mlir::PassManager& pass_manager,
                       mlir::Operation* op, llvm::raw_ostream& os) {
  std::string str;
  llvm::raw_string_ostream passOS(str);
  llvm::interleaveComma(
      pass_manager.getPasses(), passOS,
      [&](mlir::Pass& pass) { pass.printAsTextualPipeline(passOS); });
  os << "{-# external_resources: { mlir_reproducer: { pipeline: "
        "\"builtin.module("
     << passOS.str() << ")\", ";
  os << "disable_threading: true, ";
  os << "verify_each: true } } #-}";
  os << "\n\n";
}

std::string DumpMlirOpToFile(llvm::StringRef name, mlir::Operation* op,
                             llvm::StringRef dirname,
                             const mlir::PassManager* pass_manager) {
  std::unique_ptr<raw_ostream> os;
  std::string filepath;
  absl::Status result = CreateFileForDumping(name, &os, &filepath, dirname);
  if (!result.ok()) return std::string(result.message());

  LOG(INFO) << "Dumping MLIR operation '" << op->getName().getStringRef().str()
            << "' to '" << filepath << "'";
  if (pass_manager) PrintPassPipeline(*pass_manager, op, *os);
  op->print(*os, mlir::OpPrintingFlags().useLocalScope());
  return filepath;
}

std::string GetDumpDirFromEnvVar() {
  const char* prefix_env = getenv("JSV_DUMP_GRAPH_PREFIX");
  if (!prefix_env) {
    LOG(WARNING)
        << "Failed to dump MLIR module because dump location is not "
        << "specified through JSV_DUMP_GRAPH_PREFIX environment variable.";
    return "";
  }

  std::string result = prefix_env;

  if (absl::EqualsIgnoreCase(result, "sponge") &&
      !tsl::io::GetTestUndeclaredOutputsDir(&result)) {
    LOG(WARNING) << "JSV_DUMP_GRAPH_PREFIX=sponge but "
                    "TEST_UNDECLARED_OUTPUT_DIRS is not set";
    return "";
  }
  return result;
}
}  // namespace jsv
