load("//third_party/llvm/build_defs:lit.bzl", "glob_lit_tests")

package(licenses = ["notice"])

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    default_tags = ["small"],
    driver = "@neptune//runtime/core/stablehlo_transforms/mlir:run_lit.sh",
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//mlir:run_lit.sh",
        "@neptune//runtime/core/stablehlo_transforms/mlir:stablehlo-opt",
    ],
)
