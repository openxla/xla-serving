"""Python wrappers for CppAssetManager and CppResourceManager."""

import uuid

from data.resource_manager.proto import resource_manager_pb2
from data.resource_manager.python import resource_manager_pybind as ext


CppAsset = ext.CppAsset
CppResourceManager = ext.CppResourceManager
CppResource = ext.CppResource


class Asset(ext.CppAsset):
  """An asset is a descriptor of a resource, e.g. a file path."""

  def __init__(
      self, path: str, name: str | None = None, content: str | None = None
  ):
    if name is None:
      name = f'{path}_{uuid.uuid4()}'
    if content is None:
      super().__init__(name, path)
    else:
      super().__init__(name, path, content)


class ResourceManager:
  """Python wrapper for CppResourceManager."""

  def __init__(self, resources: list[CppResource]):
    self._resource_manager = ext.CppResourceManager.create(resources)

  def add_resource(self, resource: CppResource):
    self._resource_manager.add_resource(resource)

  def add_resource_ignoring_duplicates(self, resource: CppResource):
    self._resource_manager.add_resource_ignoring_duplicates(resource)

  def get_resource(self, resource_id: int) -> CppResource:
    return self._resource_manager.get_resource(int(resource_id))

  def get_all_assets(self) -> list[CppAsset]:
    return self._resource_manager.get_all_assets()

  def get_all_resources(self) -> list[CppResource]:
    return self._resource_manager.get_all_resources()

  def to_proto(self) -> resource_manager_pb2.ResourceManagerProto:
    """Serializes the ResourceManager to a proto."""
    return self._resource_manager.to_proto()

  @classmethod
  def from_proto(
      cls, proto: resource_manager_pb2.ResourceManagerProto
  ) -> 'ResourceManager':
    """Deserializes a ResourceManager from a proto.

    Args:
      proto: The proto to deserialize.

    Returns:
      A ResourceManager instance.
    """
    cpp_resource_manager = ext.CppResourceManager.from_proto(proto)
    # Create a dummy list to initialize the Python wrapper, the actual manager
    # is replaced.
    instance = cls([])
    instance._resource_manager = cpp_resource_manager
    return instance
