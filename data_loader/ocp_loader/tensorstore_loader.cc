#include "third_party/neptune/data_loader/ocp_loader/tensorstore_loader.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/log/check.h"
#include "third_party/absl/log/log.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/types/span.h"
#include "third_party/australis/google/ifrt/ifrt_australis.proto.h"
#include "third_party/australis/google/ifrt/tensorstore_loader.h"
#include "third_party/australis/google/ifrt/utils.h"
#include "third_party/json/include/nlohmann/json_fwd.hpp"
#include "third_party/neptune/data_loader/ocp_loader/common_types.h"
#include "third_party/neptune/data_loader/ocp_loader/shared.h"
#include "third_party/protobuf/repeated_ptr_field.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/array.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/device.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/memory.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/shape.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/sharding.h"
#include "third_party/tensorflow/compiler/xla/python/pjrt_ifrt/pjrt_dtype.h"
#include "third_party/tensorflow/compiler/xla/shape_util.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/errors.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"
#include "third_party/tensorflow/compiler/xla/util.h"
#include "third_party/tensorflow/compiler/xla/xla_data.proto.h"
#include "third_party/tensorstore/array.h"
#include "third_party/tensorstore/context.h"
#include "third_party/tensorstore/contiguous_layout.h"
#include "third_party/tensorstore/data_type.h"
#include "third_party/tensorstore/index.h"
#include "third_party/tensorstore/index_space/dim_expression.h"
#include "third_party/tensorstore/index_space/index_domain.h"
#include "third_party/tensorstore/open.h"
#include "third_party/tensorstore/open_mode.h"
#include "third_party/tensorstore/tensorstore.h"
#include "third_party/tensorstore/util/element_pointer.h"
#include "third_party/tensorstore/util/result.h"
#include "third_party/tf_runtime/include/tfrt/concurrency/ref_count.h"
#include "thread/fiber/contrib/status_bundle/status_bundle.h"

namespace neptune {

absl::StatusOr<tensorstore::DataType> GetTensorStoreDataTypeForPrimitiveType(
    xla::PrimitiveType type) {
  using ::tensorstore::DataType;
  using ::tensorstore::dtype_v;
  static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
                "Big endian support not implemented");
  switch (type) {
    case xla::PRED:
      return DataType(dtype_v<tensorstore::dtypes::bool_t>);
    case xla::S8:
      return DataType(dtype_v<tensorstore::dtypes::int8_t>);
    case xla::S16:
      return DataType(dtype_v<tensorstore::dtypes::int16_t>);
    case xla::S32:
      return DataType(dtype_v<tensorstore::dtypes::int32_t>);
    case xla::S64:
      return DataType(dtype_v<tensorstore::dtypes::int64_t>);
    case xla::U8:
      return DataType(dtype_v<tensorstore::dtypes::uint8_t>);
    case xla::U16:
      return DataType(dtype_v<tensorstore::dtypes::uint16_t>);
    case xla::U32:
      return DataType(dtype_v<tensorstore::dtypes::uint32_t>);
    case xla::U64:
      return DataType(dtype_v<tensorstore::dtypes::uint64_t>);
    case xla::BF16:
      return DataType(dtype_v<tensorstore::dtypes::bfloat16_t>);
    case xla::F16:
      return DataType(dtype_v<tensorstore::dtypes::float16_t>);
    case xla::F32:
      return DataType(dtype_v<tensorstore::dtypes::float32_t>);
    case xla::F64:
      return DataType(dtype_v<tensorstore::dtypes::float64_t>);
    case xla::C64:
      return DataType(dtype_v<tensorstore::dtypes::complex64_t>);
    case xla::C128:
      return DataType(dtype_v<tensorstore::dtypes::complex128_t>);
    default:
      return xla::Unimplemented("Unimplemented primitive type %s",
                                PrimitiveType_Name(type));
  }
}

absl::StatusOr<xla::PrimitiveType> TensorStoreDataTypeToXlaPrimitiveType(
    tensorstore::DataType type) {
  using ::tensorstore::DataType;
  using ::tensorstore::dtype_v;
  static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
                "Big endian support not implemented");

  if (type == DataType(dtype_v<tensorstore::dtypes::bool_t>)) {
    return xla::PRED;
  } else if (type == DataType(dtype_v<tensorstore::dtypes::int8_t>)) {
    return xla::S8;
  } else if (type == DataType(dtype_v<tensorstore::dtypes::int16_t>)) {
    return xla::S16;
  } else if (type == DataType(dtype_v<tensorstore::dtypes::int32_t>)) {
    return xla::S32;
  } else if (type == DataType(dtype_v<tensorstore::dtypes::int64_t>)) {
    return xla::S64;
  } else if (type == DataType(dtype_v<tensorstore::dtypes::uint8_t>)) {
    return xla::U8;
  } else if (type == DataType(dtype_v<tensorstore::dtypes::uint16_t>)) {
    return xla::U16;
  } else if (type == DataType(dtype_v<tensorstore::dtypes::uint32_t>)) {
    return xla::U32;
  } else if (type == DataType(dtype_v<tensorstore::dtypes::uint64_t>)) {
    return xla::U64;
  } else if (type == DataType(dtype_v<tensorstore::dtypes::bfloat16_t>)) {
    return xla::BF16;
  } else if (type == DataType(dtype_v<tensorstore::dtypes::float16_t>)) {
    return xla::F16;
  } else if (type == DataType(dtype_v<tensorstore::dtypes::float32_t>)) {
    return xla::F32;
  } else if (type == DataType(dtype_v<tensorstore::dtypes::float64_t>)) {
    return xla::F64;
  } else if (type == DataType(dtype_v<tensorstore::dtypes::complex64_t>)) {
    return xla::C64;
  } else if (type == DataType(dtype_v<tensorstore::dtypes::complex128_t>)) {
    return xla::C128;
  } else {
    return xla::Unimplemented("Unimplemented tensorstore data type %s",
                              type.name());
  }
}

absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> LoadFromTensorstore(
    xla::ifrt::Client* ifrt_client,
    const ::proto2::RepeatedPtrField<aux::TensorStoreLoadSpec>& specs,
    std::string_view base_dir, aux::LoadFromTensorstoreExtraOptions opts,
    CheckpointType checkpoint_type) {
  std::optional<tensorstore::Context> context;
  if (checkpoint_type == CheckpointType::kOcdbt) {
    TF_ASSIGN_OR_RETURN(context, GetOcdbtContext());
    auto context_spec_json = context->spec().ToJson();
    if (!context_spec_json.ok()) {
      return tensorstore::GetStatus(context_spec_json);
    }
    opts.tensorstore_context_spec = *std::move(context_spec_json);
  }

  // Do not set `base_dir` because the path in `specs` already contains the
  // base_dir.
  return aux::LoadFromTensorstore(ifrt_client, specs, /*base_dir=*/"",
                                  std::move(opts));
}

}  // namespace neptune
