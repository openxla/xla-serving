#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_OCP_LOADER_COMMON_TYPES_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_OCP_LOADER_COMMON_TYPES_H_

namespace neptune {

enum class CheckpointType {
  // In standard checkpoint mode, each param is stored in its sub directory and
  // each partition in its own file..
  kStandard,
  // OCDBT mode is an optimization for lots of small params. The param names are
  // not human-readable in the checkpoint file directory.
  // https://google.github.io/tensorstore/kvstore/ocdbt/index.html
  kOcdbt,
};

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_OCP_LOADER_COMMON_TYPES_H_
