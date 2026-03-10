#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_LIBS_MLIR_DUMP_MLIR_UTIL_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_LIBS_MLIR_DUMP_MLIR_UTIL_H_

#include <memory>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "llvm/include/llvm/ADT/StringRef.h"  // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"  // from @llvm-project

namespace jsv {

inline constexpr absl::string_view kCrashReproducerStdErr = "-";
inline constexpr absl::string_view kCrashReproducerCrashAnalysis =
    "crash_analysis";

// Creates a file to use for dumping and returns success if a file could be
// created. The opened file is placed in 'os' and the path of the file used is
// placed in 'filepath'.
//
// If the TF_DUMP_GRAPH_PREFIX environment variable is kCrashReproducerStdErr,
// then the LOG(INFO) macro is used instead.
//
// This will create a file name via prefixing `name` with the value of the
// TF_DUMP_GRAPH_PREFIX environment variable if `dirname` is empty and
// suffixing `name` with ".mlir".
absl::Status CreateFileForDumping(llvm::StringRef name,
                                  std::unique_ptr<llvm::raw_ostream>* os,
                                  std::string* filepath,
                                  llvm::StringRef dirname = "");

// Dumps MLIR operation to a file and returns the file name used.
//
// If the TF_DUMP_GRAPH_PREFIX environment variable is kCrashReproducerStdErr,
// then the MLIR operation will be logged (using the LOG(INFO) macro) instead.
//
// This will create a file name via prefixing `name` with the value of the
// TF_DUMP_GRAPH_PREFIX environment variable if `dirname` is empty and
// suffixing `name` with ".mlir".
// If `pass_manager` is provided, prints a header with the pass pipeline.
std::string DumpMlirOpToFile(llvm::StringRef name, mlir::Operation* op,
                             llvm::StringRef dirname = "",
                             const mlir::PassManager* pass_manager = nullptr);

// Reads the directory to dump the MLIR module from environment variables.
// Default is reading from TF_DUMP_GRAPH_PREFIX, and if the string is 'sponge'
// read from TEST_UNDECLARED_OUTPUTS_DIR. Returns nullptr if the directory
// cannot be determined and generates a warning message.
std::string GetDumpDirFromEnvVar();

// Prints the pass pipeline of `pass_manager` to `os`.
void PrintPassPipeline(const mlir::PassManager& pass_manager,
                       mlir::Operation* op, llvm::raw_ostream& os);
}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_LIBS_MLIR_DUMP_MLIR_UTIL_H_
