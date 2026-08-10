import ctypes
import os
from typing import cast

from absl.testing import absltest
import jax
import numpy as np

from data.processor.proto import transform_config_pb2
from data.processor.python import composite_processor_pybind
from data.processor.python import composite_processor_test_utils_pybind as test_utils
from data.processor.python.examples import add_from_resource_transform_pybind
from data.processor.python.examples import add_host_buffer_transform_pybind
from data.processor.python.examples import add_int_transform_pybind
from data.processor.python.examples import add_two_transform_pybind
from data.processor.python.examples import int_resource_pybind
from data.processor.python.examples import make_dict_transform_pybind
from data.processor.python.examples import multiply_by_five_transform_pybind
from data.processor.python.examples import suffix_transform_pybind
from data.resource_manager.python import resource_manager


class CompositeProcessorTest(absltest.TestCase):

  def test_value_creation(self):
    v_empty = composite_processor_pybind.Value()
    self.assertIsNone(v_empty.as_encapsulated_type())
    self.assertEqual(repr(v_empty), "(empty Value)")

    v_int = composite_processor_pybind.Value(10)
    self.assertEqual(v_int.as_encapsulated_type(), 10)
    self.assertEqual(repr(v_int), "Value(10)")

    v_float = composite_processor_pybind.Value(3.14)
    self.assertAlmostEqual(v_float.as_encapsulated_type(), 3.14, places=2)
    self.assertEqual(repr(v_float), "Value(3.14)")

    v_str = composite_processor_pybind.Value("hello")
    self.assertEqual(v_str.as_encapsulated_type(), "hello")
    self.assertEqual(repr(v_str), "Value(hello)")

  def test_host_buffer_creation(self):
    hb = composite_processor_pybind.HostBuffer.create_int64_array(
        [1, 2, 3, 4, 5, 6], [2, 3]
    )
    self.assertEqual(hb.dtype, np.int64)
    self.assertEqual(hb.shape, (2, 3))
    self.assertEqual(hb.storage_dtype, np.int64)

    hb = composite_processor_pybind.HostBuffer.create_int64_array(
        [1, 2, 3, 4, 5, 6]
    )
    self.assertEqual(hb.dtype, np.int64)
    self.assertEqual(hb.shape, (6,))
    self.assertEqual(hb.storage_dtype, np.int64)

  def test_host_buffer_string_array(self):
    hb = composite_processor_pybind.HostBuffer.create_string_array(
        ["a", "b", "c", "d", "e", "f"], [2, 3]
    )
    self.assertEqual(hb.shape, (2, 3))
    self.assertEqual(hb.to_string_array(), [["a", "b", "c"], ["d", "e", "f"]])

    hb = composite_processor_pybind.HostBuffer.create_string_array(
        ["a", "b", "c", "d", "e", "f"], [2, 3, 1]
    )
    self.assertEqual(hb.shape, (2, 3, 1))
    self.assertEqual(
        hb.to_string_array(),
        [[["a"], ["b"], ["c"]], [["d"], ["e"], ["f"]]],
    )

    hb = composite_processor_pybind.HostBuffer.create_string_array(
        ["a", "b", "c", "d", "e", "f"], [1, 2, 3]
    )
    self.assertEqual(hb.shape, (1, 2, 3))
    self.assertEqual(hb.to_string_array(), [[["a", "b", "c"], ["d", "e", "f"]]])

    hb = composite_processor_pybind.HostBuffer.create_string_array(
        ["a", "b", "c"]
    )
    self.assertEqual(hb.shape, (3,))
    self.assertEqual(hb.to_string_array(), ["a", "b", "c"])

    with self.assertRaisesRegex(
        ValueError, "Shape does not match the number of elements"
    ):
      composite_processor_pybind.HostBuffer.create_string_array(
          ["a", "b", "c", "d", "e", "f"], [4, 2]
      )

  def test_host_buffer_type_conversions_validation(self):
    hb_int64 = composite_processor_pybind.HostBuffer.create_int64(42)
    self.assertEqual(hb_int64.to_int64(), 42)
    with self.assertRaisesRegex(
        ValueError, "HostBuffer is not of expected type"
    ):
      hb_int64.to_string()
    with self.assertRaisesRegex(
        ValueError, "HostBuffer is not of expected type"
    ):
      hb_int64.to_int32()
    with self.assertRaisesRegex(
        ValueError, "HostBuffer is not of expected type"
    ):
      hb_int64.to_string_array()

    hb_str = composite_processor_pybind.HostBuffer.create_string("hello")
    self.assertEqual(hb_str.to_string(), "hello")
    with self.assertRaisesRegex(
        ValueError, "HostBuffer is not of expected type"
    ):
      hb_str.to_int64()
    with self.assertRaisesRegex(
        ValueError, "HostBuffer is not of expected type"
    ):
      hb_str.to_int32()

  def test_host_buffer_create_from_bytes_object_ndarray(self):
    # Success case
    arr = np.array([b"a", b"b", b"c"], dtype=object)
    hb = composite_processor_pybind.HostBuffer.create_from_bytes_object_ndarray(
        arr
    )
    self.assertEqual(hb.shape, (3,))
    self.assertEqual(hb.to_string_array(), ["a", "b", "c"])

  def test_host_buffer_create_from_bytes_object_ndarray_non_contiguous(self):
    # Non-contiguous array (sliced)
    arr = np.array(
        [b"a", b"placeholder", b"b", b"placeholder", b"c"], dtype=object
    )
    non_contiguous_arr = arr[::2]
    # Verify it is indeed non-contiguous
    self.assertFalse(non_contiguous_arr.flags["C_CONTIGUOUS"])
    hb = composite_processor_pybind.HostBuffer.create_from_bytes_object_ndarray(
        non_contiguous_arr
    )
    self.assertEqual(hb.shape, (3,))
    self.assertEqual(hb.to_string_array(), ["a", "b", "c"])

  def test_host_buffer_create_from_bytes_object_ndarray_non_contiguous_2d(self):
    # Larger 2D non-contiguous array (sliced with different step in dimensions)
    arr = np.array(
        [
            [b"a00", b"xxx", b"xxx", b"a01", b"xxx", b"xxx"],
            [b"xxx", b"xxx", b"xxx", b"xxx", b"xxx", b"xxx"],
            [b"a10", b"xxx", b"xxx", b"a11", b"xxx", b"xxx"],
            [b"xxx", b"xxx", b"xxx", b"xxx", b"xxx", b"xxx"],
            [b"a20", b"xxx", b"xxx", b"a21", b"xxx", b"xxx"],
            [b"xxx", b"xxx", b"xxx", b"xxx", b"xxx", b"xxx"],
        ],
        dtype=object,
    )
    non_contiguous_arr = arr[::2, ::3]  # should be shape (3, 2)
    self.assertFalse(non_contiguous_arr.flags["C_CONTIGUOUS"])
    self.assertFalse(non_contiguous_arr.flags["F_CONTIGUOUS"])
    hb = composite_processor_pybind.HostBuffer.create_from_bytes_object_ndarray(
        non_contiguous_arr
    )
    self.assertEqual(hb.shape, (3, 2))
    self.assertEqual(
        hb.to_string_array(),
        [["a00", "a01"], ["a10", "a11"], ["a20", "a21"]],
    )

  def test_host_buffer_create_from_bytes_object_ndarray_negative_strided_buffer(
      self,
  ):
    original_arr = np.zeros(500, dtype=np.uint8)
    m_view = memoryview(original_arr[::-1])
    obj_array = np.empty(1, dtype=object)
    obj_array[0] = m_view
    with self.assertRaisesRegex(
        ValueError, "Buffer object must be C-contiguous"
    ):
      composite_processor_pybind.HostBuffer.create_from_bytes_object_ndarray(
          obj_array
      )

  def test_host_buffer_create_from_bytes_object_ndarray_multibyte_buffer(self):
    buf = np.array([0x61626364], dtype=np.uint32)
    obj_array = np.empty(1, dtype=object)
    obj_array[0] = buf
    hb = composite_processor_pybind.HostBuffer.create_from_bytes_object_ndarray(
        obj_array
    )
    self.assertEqual(hb.shape, (1,))
    res = hb.to_string_array()
    self.assertLen(res[0], 4)

  def test_host_buffer_create_from_bytes_object_ndarray_wrong_dtype(self):
    # Failure case: non-object array
    arr_int = np.array([1, 2, 3])
    with self.assertRaisesRegex(ValueError, "Expected an object array"):
      composite_processor_pybind.HostBuffer.create_from_bytes_object_ndarray(
          arr_int
      )

  def test_host_buffer_create_from_bytes_object_ndarray_supported_types(self):
    # Mixed types: bytes, str, None, and other buffer-like (e.g. bytearray,
    # memoryview)
    arr = np.array(
        [b"a", "b", None, bytearray(b"d"), memoryview(b"e")], dtype=object
    )
    hb = composite_processor_pybind.HostBuffer.create_from_bytes_object_ndarray(
        arr
    )
    self.assertEqual(hb.shape, (5,))
    self.assertEqual(hb.to_string_array(), ["a", "b", "", "d", "e"])

  def test_host_buffer_create_from_bytes_object_ndarray_unsupported_element_type(
      self,
  ):
    # Failure case: object array with int
    arr_int = np.array([b"a", 1, b"c"], dtype=object)
    with self.assertRaisesRegex(
        ValueError, "Elements must be str, buffer, or None, got int"
    ):
      composite_processor_pybind.HostBuffer.create_from_bytes_object_ndarray(
          arr_int
      )

  def test_execution_graph_and_processor(self):
    graph = composite_processor_pybind.ExecutionGraph()

    t_add = add_int_transform_pybind.AddIntTransform()

    idx = graph.add_node(
        "my_add",
        t_add,
        [
            composite_processor_pybind.GraphEdge.graph_input(0),
            composite_processor_pybind.GraphEdge.graph_input(1),
        ],
    )
    graph.set_outputs(
        [composite_processor_pybind.GraphEdge.node_output(idx, 0)]
    )

    processor = composite_processor_pybind.CompositeProcessor.create(graph)

    inputs = [
        composite_processor_pybind.Value(5),
        composite_processor_pybind.Value(10),
    ]
    outputs = processor.process(inputs)

    self.assertLen(outputs, 1)
    self.assertEqual(outputs[0].as_encapsulated_type(), 15)

  def test_execute_context(self):
    context = composite_processor_pybind.ExecuteContext()
    inputs = [
        composite_processor_pybind.Value(1),
        composite_processor_pybind.Value(2),
    ]
    context.inputs = inputs
    self.assertLen(context.inputs, 2)
    self.assertEqual(context.inputs[0].as_encapsulated_type(), 1)
    self.assertEqual(context.inputs[1].as_encapsulated_type(), 2)

  def test_add_transform_internal_api(self):
    transform = add_int_transform_pybind.AddIntTransform()
    context = composite_processor_pybind.ExecuteContext()
    context.inputs = [
        composite_processor_pybind.Value(10),
        composite_processor_pybind.Value(20),
    ]
    transform.execute(context)
    self.assertLen(context.outputs, 1)
    self.assertEqual(context.outputs[0].as_encapsulated_type(), 30)

  def test_add_transform(self):
    transform = add_int_transform_pybind.AddIntTransform()
    output = transform.execute(10, 20)
    self.assertEqual(output, 30)

  def test_add_from_resource_transform(self):
    # Instantiate resources.
    tmp_path = os.path.join(self.create_tempdir().full_path, "asset1.txt")
    with open(tmp_path, "w") as f:
      f.write("42")
    asset = resource_manager.Asset(path=tmp_path, name="asset1")
    resource = int_resource_pybind.IntResource([asset])
    resource.load()
    rm = resource_manager.ResourceManager([resource])  # pylint: disable=unused-variable

    # Instantiate transform.
    transform = add_from_resource_transform_pybind.AddFromResourceTransform(
        resource
    )
    graph = composite_processor_pybind.ExecutionGraph()
    idx = graph.add_node(
        "add_from_resource",
        transform,
        [composite_processor_pybind.GraphEdge.graph_input(0)],
    )
    graph.set_outputs(
        [composite_processor_pybind.GraphEdge.node_output(idx, 0)]
    )
    processor = composite_processor_pybind.CompositeProcessor.create(graph)

    # Verify execution.
    inputs = [composite_processor_pybind.Value(100)]
    outputs = processor.process(inputs)
    self.assertLen(outputs, 1)
    self.assertEqual(outputs[0].as_encapsulated_type(), 142)

    # Serialize the transform.
    config = processor.to_proto()

    # Reload the processor using the config
    processor = composite_processor_pybind.CompositeProcessor.from_proto(config)

    # Verify execution result remains consistent with the serialized resource.
    inputs = [composite_processor_pybind.Value(200)]
    outputs = processor.process(inputs)
    self.assertLen(outputs, 1)
    self.assertEqual(outputs[0].as_encapsulated_type(), 242)

  def test_suffix_transform(self):
    transform = suffix_transform_pybind.SuffixTransform("_is_great")
    output = transform.execute("voxel")
    self.assertEqual(output, "voxel_is_great")

  def test_make_dict_transform(self):
    transform = make_dict_transform_pybind.MakeDictTransform()
    output = transform.execute("key", 42)
    self.assertEqual(output, {"key": 42})

  def test_multiply_by_five_transform(self):
    transform = multiply_by_five_transform_pybind.MultiplyByFiveTransform()
    output = transform.execute(10)
    self.assertEqual(output, 50)

  def test_add_two_transform(self):
    transform = add_two_transform_pybind.AddTwoTransform()
    output = transform.execute(7)
    self.assertEqual(output, 9)

  def test_add_host_buffer_transform(self):
    transform = add_host_buffer_transform_pybind.AddHostBufferTransform()
    in_buffer1 = composite_processor_pybind.HostBuffer.create_int32(5)
    in_buffer2 = composite_processor_pybind.HostBuffer.create_int32(10)
    output = transform.execute(in_buffer1, in_buffer2)
    self.assertEqual(output.to_int32(), 15)

  def test_add_host_buffer_vector_transform(self):
    transform = add_host_buffer_transform_pybind.AddHostBufferVectorTransform()
    in_buffer1 = composite_processor_pybind.HostBuffer.create_int32(5)
    in_buffer2 = composite_processor_pybind.HostBuffer.create_int32(10)
    in_buffer3 = composite_processor_pybind.HostBuffer.create_int32(20)
    output = transform.execute([in_buffer1, in_buffer2, in_buffer3])
    self.assertEqual(output.to_int32(), 35)

    context = composite_processor_pybind.ExecuteContext()
    context.inputs = [
        composite_processor_pybind.Value([in_buffer1, in_buffer2, in_buffer3])
    ]
    transform.execute(context)
    self.assertLen(context.outputs, 1)
    out_hb = cast(
        composite_processor_pybind.HostBuffer,
        context.outputs[0].as_encapsulated_type(),
    )
    self.assertEqual(out_hb.to_int32(), 35)

  def test_unsupported_dtype(self):
    transform = add_host_buffer_transform_pybind.AddHostBufferVectorTransform()
    in_buffer_str = composite_processor_pybind.HostBuffer.create_string("hello")
    context = composite_processor_pybind.ExecuteContext()
    context.inputs = [composite_processor_pybind.Value([in_buffer_str])]
    with self.assertRaisesRegex(RuntimeError, "unsupported dtype"):
      transform.execute(context)

  def test_host_buffer_properties(self):
    hb = composite_processor_pybind.HostBuffer.create_int64(100)
    self.assertEqual(hb.dtype, np.int64)
    self.assertEqual(hb.shape, (1,))
    self.assertEqual(hb.storage_dtype, np.int64)
    self.assertEqual(repr(hb), "<HostBuffer: dtype=S64 shape=[1]>")

    hb = composite_processor_pybind.HostBuffer.create_string("hello")
    self.assertEqual(hb.dtype, np.object_)
    self.assertEqual(hb.shape, (1,))
    self.assertEqual(hb.storage_dtype, np.object_)
    self.assertEqual(repr(hb), "<HostBuffer: dtype=STRING shape=[1]>")

    val = composite_processor_pybind.Value(hb)
    self.assertRegex(
        repr(val),
        r"Value\(<Object of type .*HostBuffer.*>\)",
    )

  def test_create_bytes_array_from_bytes(self):
    # Dynamically construct each bytes object to prevent referencing
    # constant objects which will remain alive even after the
    # container bytes object is deallocated.
    vals = [b"hello_" + str(i).encode() for i in range(2)]
    shape = [2, 1]
    hb = composite_processor_pybind.HostBuffer.create_bytes_array(vals, shape)
    # Delete the list to verify the HostBuffer keeps the strings alive.
    del vals
    arr = hb.to_string_array()
    self.assertEqual(arr, [["hello_0"], ["hello_1"]])

  def test_create_bytes_array_delete_individual_obj(self):
    # Dynamically construct each bytes object to prevent referencing
    # constant objects which will remain alive even after the
    # container bytes object is deallocated.
    vals = [b"hello_" + str(i).encode() for i in range(2)]
    shape = [2, 1]
    hb = composite_processor_pybind.HostBuffer.create_bytes_array(vals, shape)
    # Delete an item in the list to verify the HostBuffer keeps the
    # strings alive.
    del vals[0]
    arr = hb.to_string_array()
    self.assertEqual(arr, [["hello_0"], ["hello_1"]])

  def test_create_bytes_array_from_bytearray_mutation_prevented(self):
    b = bytearray(b"initial_payload")
    hb = composite_processor_pybind.HostBuffer.create_bytes_array([b], [1])
    self.assertEqual(hb.to_string_array(), ["initial_payload"])
    with self.assertRaises(BufferError):
      b.extend(b"mutation")
    # Also verify that the buffer still has the correct content
    self.assertEqual(hb.to_string_array(), ["initial_payload"])

  def test_create_from_bytes_object_ndarray_from_bytearray_mutation_prevented(
      self,
  ):
    b = bytearray(b"initial_payload_2")
    arr = np.empty(1, dtype=object)
    arr[0] = b
    hb = composite_processor_pybind.HostBuffer.create_from_bytes_object_ndarray(
        arr
    )
    self.assertEqual(hb.to_string_array(), ["initial_payload_2"])
    with self.assertRaises(BufferError):
      b.clear()
    self.assertEqual(hb.to_string_array(), ["initial_payload_2"])

  def test_bytearray_mutation_allowed_after_host_buffer_deletion(self):
    b = bytearray(b"initial")
    hb = composite_processor_pybind.HostBuffer.create_bytes_array([b], [1])
    self.assertEqual(hb.to_string_array(), ["initial"])
    with self.assertRaises(BufferError):
      b.clear()

    # Delete HostBuffer
    del hb

    # Now it should be allowed
    b.clear()
    self.assertEqual(b, bytearray(b""))

  def test_host_buffer_from_jax_dlpack(self):
    jax.config.update("jax_enable_x64", True)
    ja = jax.numpy.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    hb = composite_processor_pybind.HostBuffer.from_dlpack(ja)
    self.assertEqual(hb.dtype, np.int32)
    self.assertEqual(hb.shape, (2, 3))

    na_out = np.from_dlpack(hb).copy()
    self.assertEqual(na_out.shape, (2, 3))
    self.assertEqual(na_out.dtype, np.int32)
    self.assertEqual(na_out.tolist(), [[1, 2, 3], [4, 5, 6]])

  def test_host_buffer_from_numpy_dlpack(self):
    na = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    hb = composite_processor_pybind.HostBuffer.from_dlpack(na)
    self.assertEqual(hb.dtype, np.int32)
    self.assertEqual(hb.shape, (2, 3))

    na_out = np.from_dlpack(hb).copy()
    self.assertEqual(na_out.shape, (2, 3))
    self.assertEqual(na_out.dtype, np.int32)
    self.assertEqual(na_out.tolist(), [[1, 2, 3], [4, 5, 6]])

  def test_host_buffer_from_jax_dlpack_copy(self):
    jax.config.update("jax_enable_x64", True)
    # JAX array data buffers are always Eigen-aligned on CPU.
    ja = jax.numpy.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    hb = composite_processor_pybind.HostBuffer.from_dlpack(ja, copy=True)
    self.assertEqual(hb.dtype, np.int32)
    self.assertEqual(hb.shape, (2, 3))

    na_out = np.from_dlpack(hb).copy()
    self.assertEqual(na_out.tolist(), [[1, 2, 3], [4, 5, 6]])

    hb = composite_processor_pybind.HostBuffer.from_dlpack(ja, copy=False)
    # JAX arrays are immutable, so we cannot use the usual way of modifying
    # the original array to verify it is not a copy.
    # For now, just verify the contents.
    self.assertEqual(hb.dtype, np.int32)
    self.assertEqual(hb.shape, (2, 3))
    na_out = np.from_dlpack(hb).copy()
    self.assertEqual(na_out.tolist(), [[1, 2, 3], [4, 5, 6]])

  def test_host_buffer_from_numpy_dlpack_copy(self):
    # Re-create until the array is Eigen-aligned.
    na = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    while not test_utils.is_numpy_eigen_aligned(na):
      na = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    hb = composite_processor_pybind.HostBuffer.from_dlpack(na, copy=True)
    self.assertEqual(hb.dtype, np.int32)
    self.assertEqual(hb.shape, (2, 3))

    na_out = np.from_dlpack(hb).copy()
    self.assertEqual(na_out.tolist(), [[1, 2, 3], [4, 5, 6]])

    hb = composite_processor_pybind.HostBuffer.from_dlpack(na, copy=False)
    self.assertEqual(hb.dtype, np.int32)
    self.assertEqual(hb.shape, (2, 3))
    # Try changing the original array to verify it is not a copy.
    na[0, 0] = 100
    # HostBuffer does not have a way to index into an array, but the scalar test
    # methods will return the first element, so we use this as a workaround.
    hb_0_0 = hb.to_int32()
    self.assertEqual(hb_0_0, 100)

  def test_host_buffer_from_numpy_noncontiguous_copy_raises(self):
    na = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    na_noncontiguous = np.transpose(na)
    with self.assertRaises(ValueError):
      composite_processor_pybind.HostBuffer.from_dlpack(
          na_noncontiguous, copy=False
      )

  def test_host_buffer_from_dlpack_invalid_device(self):
    class InvalidDeviceDLPack:

      def __dlpack__(self):
        return None

      def __dlpack_device__(self):
        # 2 is kDLCUDA in DLPack
        return (2, 0)

    with self.assertRaises(BufferError):
      composite_processor_pybind.HostBuffer.from_dlpack(InvalidDeviceDLPack())

  def test_host_buffer_from_dlpack_renames_capsule(self):
    jax.config.update("jax_enable_x64", True)
    ja = jax.numpy.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    # Call __dlpack__ explicitly to get the capsule
    dlpack_capsule = ja.__dlpack__()

    class ReusedCapsuleDLPack:

      def __dlpack__(self):
        return dlpack_capsule

      def __dlpack_device__(self):
        return (1, 0)  # kDLCPU

    reused_obj = ReusedCapsuleDLPack()

    # First consumption should succeed
    hb = composite_processor_pybind.HostBuffer.from_dlpack(reused_obj)
    self.assertEqual(hb.shape, (2, 3))

    # Second consumption of the same capsule should fail because it was renamed
    with self.assertRaisesRegex(ValueError, "Invalid DLPack capsule name"):
      composite_processor_pybind.HostBuffer.from_dlpack(reused_obj)

  def test_host_buffer_from_dlpack_unnamed_capsule(self):
    ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
    ctypes.pythonapi.PyCapsule_New.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
    ]

    class UnnamedCapsuleDLPack:

      def __dlpack__(self):
        return ctypes.pythonapi.PyCapsule_New(ctypes.c_void_p(1), None, None)

      def __dlpack_device__(self):
        return (1, 0)  # kDLCPU

    with self.assertRaisesRegex(ValueError, "Capsule has no name"):
      composite_processor_pybind.HostBuffer.from_dlpack(UnnamedCapsuleDLPack())

  def test_host_buffer_from_dlpack_malformed_bool(self):
    # Define DLPack structures using ctypes
    class DLDevice(ctypes.Structure):
      _fields_ = [
          ("device_type", ctypes.c_uint),
          ("device_id", ctypes.c_int32),
      ]

    class DLDataType(ctypes.Structure):
      _fields_ = [
          ("code", ctypes.c_uint8),
          ("bits", ctypes.c_uint8),
          ("lanes", ctypes.c_uint16),
      ]

    class DLTensor(ctypes.Structure):
      _fields_ = [
          ("data", ctypes.c_void_p),
          ("device", DLDevice),
          ("ndim", ctypes.c_int32),
          ("dtype", DLDataType),
          ("shape", ctypes.POINTER(ctypes.c_int64)),
          ("strides", ctypes.POINTER(ctypes.c_int64)),
          ("byte_offset", ctypes.c_uint64),
      ]

    DLManagedTensorDeleter = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

    class DLManagedTensor(ctypes.Structure):
      _fields_ = [
          ("dl_tensor", DLTensor),
          ("manager_ctx", ctypes.c_void_p),
          ("deleter", DLManagedTensorDeleter),
      ]

    # Create a malformed DLManagedTensor: kDLBool (6) with 1 bit
    data = (ctypes.c_bool * 1)(True)
    shape = (ctypes.c_int64 * 1)(1)

    dl_tensor = DLTensor(
        data=ctypes.cast(data, ctypes.c_void_p),
        device=DLDevice(device_type=1, device_id=0),  # kDLCPU
        ndim=1,
        dtype=DLDataType(code=6, bits=1, lanes=1),  # kDLBool, 1 bit
        shape=ctypes.cast(shape, ctypes.POINTER(ctypes.c_int64)),
        strides=None,
        byte_offset=0,
    )

    @DLManagedTensorDeleter
    def deleter(ptr):
      pass

    managed_tensor = DLManagedTensor(
        dl_tensor=dl_tensor, manager_ctx=None, deleter=deleter
    )

    # Wrap in PyCapsule
    ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
    ctypes.pythonapi.PyCapsule_New.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
    ]

    capsule_name = b"dltensor"
    managed_tensor_ptr = ctypes.pointer(managed_tensor)
    capsule = ctypes.pythonapi.PyCapsule_New(
        ctypes.cast(managed_tensor_ptr, ctypes.c_void_p), capsule_name, None
    )

    class MalformedBoolDLPack:

      def __dlpack__(self):
        return capsule

      def __dlpack_device__(self):
        return (1, 0)  # kDLCPU

    with self.assertRaisesRegex(
        ValueError, "Unsupported kDLBool bits: 1. Expected 8 bits."
    ):
      composite_processor_pybind.HostBuffer.from_dlpack(MalformedBoolDLPack())

  def test_host_buffer_from_dlpack_strided(self):
    jax.config.update("jax_enable_x64", True)
    # Create a strided array (transposed)
    ja = jax.numpy.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32).T
    self.assertEqual(ja.shape, (3, 2))
    # Pass the strided array via DLPack
    hb = composite_processor_pybind.HostBuffer.from_dlpack(ja)
    self.assertEqual(hb.dtype, np.int32)
    self.assertEqual(hb.shape, (3, 2))

    na_out = np.from_dlpack(hb).copy()
    self.assertEqual(na_out.shape, (3, 2))
    self.assertEqual(na_out.dtype, np.int32)
    self.assertEqual(na_out.tolist(), [[1, 4], [2, 5], [3, 6]])

  def test_dlpack_interface(self):
    jax.config.update("jax_enable_x64", True)
    hb = composite_processor_pybind.HostBuffer.create_int32(12345)
    ja: jax.Array = jax.dlpack.from_dlpack(hb)
    self.assertEqual(ja.shape, (1,))
    self.assertEqual(ja.dtype, np.int32)
    self.assertEqual(ja[0], 12345)

    hb = composite_processor_pybind.HostBuffer.create_int64_array(
        [1, 2, 3, 4, 5, 6], [2, 3]
    )
    ja: jax.Array = jax.dlpack.from_dlpack(hb)
    self.assertEqual(ja.shape, (2, 3))
    self.assertEqual(ja.dtype, np.int64)
    self.assertEqual(ja.tolist(), [[1, 2, 3], [4, 5, 6]])
    self.assertEqual(ja[1][2], 6)  # access by index works too.

    # Convert to Numpy via DLPack to explicitly test the strides.
    # For a 2x3 int64 array (8 bytes per element), C-contiguous row-major
    # expected byte strides are (3 * 8 = 24, 1 * 8 = 8).
    na = np.from_dlpack(hb)
    self.assertEqual(na.strides, (24, 8))
    self.assertEqual(na.shape, (2, 3))
    self.assertEqual(na.dtype, np.int64)
    self.assertEqual(na.tolist(), [[1, 2, 3], [4, 5, 6]])
    self.assertEqual(na[1][1], 5)  # access by index works too.

    # Try a higher version of DLPack, for DLManagedTensorVersioned.
    class HigherVersionDLPack:

      def __init__(self, host_buffer: composite_processor_pybind.HostBuffer):
        self.host_buffer = host_buffer

      def __dlpack__(
          self, stream=None, max_version=None, dl_device=None, copy=None
      ):
        del max_version
        return self.host_buffer.__dlpack__(
            stream, max_version=(1, 0), dl_device=dl_device, copy=copy
        )

    hvhb = HigherVersionDLPack(hb)
    na = np.from_dlpack(hvhb)
    self.assertEqual(na.strides, (24, 8))
    self.assertEqual(na.shape, (2, 3))
    self.assertEqual(na.dtype, np.int64)
    self.assertEqual(na.tolist(), [[1, 2, 3], [4, 5, 6]])
    self.assertEqual(na[0][2], 3)  # access by index works too.

  def test_host_buffer_pass_by_reference_no_copy(self):
    hb = composite_processor_pybind.HostBuffer.create_int64(100)
    initial_ref_count = hb.get_ref_count()
    ref_count_from_ref = test_utils.get_ref_count_from_ref(hb)
    self.assertEqual(ref_count_from_ref, initial_ref_count)

  def test_python_transform_trampoline(self):
    class MyPythonTransform(composite_processor_pybind.TransformBase):

      def __init__(self, name, input_types, output_types=None):
        super().__init__()
        self._name = name
        self._input_types = input_types
        self._output_types = output_types or []

      def DebugString(self) -> str:
        return f"DynamicName({self._name})"

      def InputTypes(self) -> list[composite_processor_pybind.TypeInfo]:
        return self._input_types

      def OutputTypes(self) -> list[composite_processor_pybind.TypeInfo]:
        return self._output_types

      def Execute(self, context) -> None:
        outputs = []
        for val in context.inputs:
          encap = val.as_encapsulated_type()
          if isinstance(encap, int):
            outputs.append(composite_processor_pybind.Value(encap + 1))
          elif isinstance(encap, composite_processor_pybind.HostBuffer):
            arr = np.from_dlpack(encap)
            new_arr = arr + 1
            new_hb = composite_processor_pybind.HostBuffer.from_dlpack(new_arr)
            outputs.append(composite_processor_pybind.Value(new_hb))
          elif isinstance(encap, str):
            outputs.append(
                composite_processor_pybind.Value(encap + "_modified")
            )
          else:
            outputs.append(val)
        context.outputs = outputs

      def ToProto(self) -> transform_config_pb2.TransformConfig:
        config = transform_config_pb2.TransformConfig()
        config.name = self._name
        config.op_name = "MyPythonTransform"
        return config

    type_info = composite_processor_pybind.TypeInfo.host_buffer()
    t = MyPythonTransform("test", [type_info], [type_info])

    # Test DebugString (repr calls DebugString)
    self.assertEqual(repr(t), "DynamicName(test)")

    # Test InputTypes (calls InputTypes and converts to list of TypeInfo)
    input_types = t.input_types()
    self.assertLen(input_types, 1)
    self.assertEqual(input_types[0], type_info)

    # Test OutputTypes
    output_types = t.output_types()
    self.assertLen(output_types, 1)
    self.assertEqual(output_types[0], type_info)

    # Test ToProto
    proto = t.to_proto()
    self.assertEqual(proto.name, "test")
    self.assertEqual(proto.op_name, "MyPythonTransform")

    # Test Execute with HostBuffer
    hb = composite_processor_pybind.HostBuffer.create_int64(10)
    val = composite_processor_pybind.Value(hb)
    context = composite_processor_pybind.ExecuteContext()
    context.inputs = [val]

    t.execute(context)
    self.assertLen(context.outputs, 1)
    out_hb = cast(
        composite_processor_pybind.HostBuffer,
        context.outputs[0].as_encapsulated_type(),
    )
    self.assertEqual(out_hb.to_int64(), 11)

  def test_python_transform_trampoline_concurrent(self):
    class MyConcurrentTransform(composite_processor_pybind.TransformBase):

      def __init__(self):
        super().__init__()
        self._input_types = [composite_processor_pybind.TypeInfo.host_buffer()]

      def DebugString(self) -> str:
        return "ConcurrentTransform"

      def InputTypes(self) -> list[composite_processor_pybind.TypeInfo]:
        return self._input_types

      def OutputTypes(self) -> list[composite_processor_pybind.TypeInfo]:
        return []

      def Execute(self, context) -> None:
        pass

      def ToProto(self) -> transform_config_pb2.TransformConfig:
        return transform_config_pb2.TransformConfig()

    t = MyConcurrentTransform()
    test_utils.run_transform_methods_concurrently(t)

  def test_dag_execution(self):
    graph = composite_processor_pybind.ExecutionGraph()

    t_add_two_1 = add_two_transform_pybind.AddTwoTransform()
    t_add_two_2 = add_two_transform_pybind.AddTwoTransform()
    t_add_int = add_int_transform_pybind.AddIntTransform()

    idx_1 = graph.add_node(
        "add_two_1",
        t_add_two_1,
        [composite_processor_pybind.GraphEdge.graph_input(0)],
    )
    idx_2 = graph.add_node(
        "add_two_2",
        t_add_two_2,
        [composite_processor_pybind.GraphEdge.graph_input(0)],
    )

    idx_3 = graph.add_node(
        "add_int",
        t_add_int,
        [
            composite_processor_pybind.GraphEdge.node_output(idx_1, 0),
            composite_processor_pybind.GraphEdge.node_output(idx_2, 0),
        ],
    )

    graph.set_outputs(
        [composite_processor_pybind.GraphEdge.node_output(idx_3, 0)]
    )

    processor = composite_processor_pybind.CompositeProcessor.create(graph)

    inputs = [composite_processor_pybind.Value(5)]
    outputs = processor.process(inputs)

    self.assertLen(outputs, 1)
    self.assertEqual(outputs[0].as_encapsulated_type(), 14)

  def test_edge_validation(self):
    with self.assertRaisesRegex(
        ValueError, "GraphInput index must be non-negative"
    ):
      composite_processor_pybind.GraphEdge.graph_input(-1)

    with self.assertRaisesRegex(
        ValueError, "NodeOutput node_index must be non-negative"
    ):
      composite_processor_pybind.GraphEdge.node_output(-1, 0)

    with self.assertRaisesRegex(
        ValueError, "NodeOutput port_index must be non-negative"
    ):
      composite_processor_pybind.GraphEdge.node_output(0, -1)


if __name__ == "__main__":
  absltest.main()
