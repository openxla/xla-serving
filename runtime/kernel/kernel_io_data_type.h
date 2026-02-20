#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_KERNEL_IO_DATA_TYPE_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_KERNEL_IO_DATA_TYPE_H_

namespace jsv {

// The data type of the kernel input and output. This is the data type wrapped
// by `jsv::Value`.
// TODO: build a type checker around this and use it in kernel
// implementations.
enum class KernelIODataType {
  kHostBuffer,  // `jsv::HostBuffer`
  kTensor,      // `tensorflow::Tensor`
};
}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_KERNEL_IO_DATA_TYPE_H_
