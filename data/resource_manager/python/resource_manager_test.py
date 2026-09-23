# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl import logging
from absl.testing import absltest
from google.protobuf import text_format

from data.resource_manager.proto import resource_manager_pb2
from data.resource_manager.python import resource_manager
from data.resource_manager.python.examples import another_concrete_resource_pybind
from data.resource_manager.python.examples import concrete_resource_pybind
from data.test_utils import test_utils
from pybind11_abseil import status as error

_STATUS_NOT_OK = (error.StatusNotOk, RuntimeError)

_TEST_PROTO_MESSAGE = """
  resources {{
    id: {id1}
    name: "{res1}"
    assets {{ name: "asset1" path: "/path/to/asset1" }}
    assets {{ name: "asset2" path: "/path/to/asset2" }}
    class_name: "CppConcreteResource"
  }}
  resources {{
    id: {id2}
    name: "{res2}"
    assets {{ name: "asset2" path: "/path/to/asset2" }}
    class_name: "OkConcreteResource"
  }}
  resources {{
    id: {id3}
    name: "{res3}"
    assets {{ name: "asset3" path: "/path/to/asset3" }}
    class_name: "AnotherConcreteResource"
  }}
  resources {{
    id: {id4}
    name: "{res4}"
    class_name: "AnotherConcreteResource"
  }}
"""


class FakeResource(resource_manager.CppResource):

  def __init__(self, resource_id=None, assets=None):
    assets_to_pass = []
    if assets is not None:
      assets_to_pass = assets
    if resource_id is None:
      super().__init__(assets_to_pass)
    else:
      super().__init__(resource_id, "", assets_to_pass, "FakeResource")
    self.loaded = False

  def LoadImpl(self):
    logging.warn(
        "NOTE: `LoadImpl` definitions in python are not serializable."
        " Trampoline is provided for testing only."
    )
    self.loaded = True


class ResourceManagerTest(absltest.TestCase):

  def test_asset_init(self):
    asset = resource_manager.Asset(path="/path/to/asset1", name="asset1")
    self.assertEqual(asset.name, "asset1")
    self.assertEqual(asset.path, "/path/to/asset1")

  def test_cpp_asset_init_with_content(self):
    asset = resource_manager.Asset("/path/to/asset1", "asset1", "some content")
    self.assertEqual(asset.name, "asset1")
    self.assertEqual(asset.path, "/path/to/asset1")
    self.assertEqual(asset.content, "some content")

  def test_asset_load_content_not_found(self):
    asset = resource_manager.Asset(path="/no/such/asset2", name="asset2")
    with self.assertRaisesRegex(_STATUS_NOT_OK, "NOT_FOUND"):
      asset.load_content()

  def test_resource_manager_create_duplicate_resource_id_is_error(self):
    resource1 = FakeResource(resource_id=1)
    resource2 = FakeResource(resource_id=1)
    with self.assertRaisesRegex(_STATUS_NOT_OK, "ALREADY_EXISTS"):
      resource_manager.ResourceManager([resource1, resource2])

  def test_resource_manager_create_duplicate_resource_is_error(self):
    resource = FakeResource()
    with self.assertRaisesRegex(_STATUS_NOT_OK, "ALREADY_EXISTS"):
      resource_manager.ResourceManager([resource, resource])

  def test_resource_manager_add_resource_ignoring_duplicates_and_get(self):
    rm = resource_manager.ResourceManager([])
    res = FakeResource()
    rm.add_resource_ignoring_duplicates(res)
    # Duplicate calls should not throw an error.
    rm.add_resource_ignoring_duplicates(res)
    # Duplicate calls with same id and different object should not overwrite.
    rm.add_resource_ignoring_duplicates(FakeResource(resource_id=res.id))
    got = rm.get_resource(res.id)
    self.assertIsInstance(got, resource_manager.CppResource)
    self.assertIs(res, got)

  def test_resource_manager_add_resource_and_get(self):
    rm = resource_manager.ResourceManager([])
    res = FakeResource()
    rm.add_resource(res)
    got = rm.get_resource(res.id)
    self.assertIsInstance(got, resource_manager.CppResource)
    self.assertIs(res, got)

  def test_resource_manager_add_resource_duplicate_is_error(self):
    rm = resource_manager.ResourceManager([])
    res = FakeResource()
    rm.add_resource(res)
    with self.assertRaisesRegex(_STATUS_NOT_OK, "ALREADY_EXISTS"):
      rm.add_resource(res)

  def test_resource_manager_get_nonexistent(self):
    rm = resource_manager.ResourceManager([])
    with self.assertRaisesRegex(_STATUS_NOT_OK, "NOT_FOUND"):
      rm.get_resource(999)

  def test_resource_manager_get_all_assets(self):
    asset1 = resource_manager.Asset(path="/path/to/asset1", name="asset1")
    asset2 = resource_manager.Asset(path="/path/to/asset2", name="asset2")
    asset3 = resource_manager.Asset(path="/path/to/asset3", name="asset3")
    asset4 = resource_manager.Asset(path="/path/to/asset4", name="asset4")
    res = FakeResource(assets=[asset1, asset2])
    res2 = FakeResource(assets=[asset3])
    res3 = FakeResource(assets=[asset4])
    rm = resource_manager.ResourceManager([res, res2])
    rm.add_resource_ignoring_duplicates(res3)
    assets = rm.get_all_assets()
    self.assertEqual(len(assets), 4)
    self.assertEqual(
        sorted([asset.name for asset in assets]),
        ["asset1", "asset2", "asset3", "asset4"],
    )

  def test_resource_manager_get_all_resources(self):
    res1 = FakeResource(assets=[resource_manager.Asset("p1", "a1")])
    res2 = FakeResource(assets=[resource_manager.Asset("p2", "a2")])
    rm = resource_manager.ResourceManager([res1, res2])
    resources = rm.get_all_resources()
    self.assertEqual(len(resources), 2)
    names = {r.name for r in resources}
    self.assertEqual(names, {res1.name, res2.name})

  def test_auto_generated_resource_pybinding(self):
    asset1 = resource_manager.Asset(path="/path/to/asset1", name="asset1")
    asset2 = resource_manager.Asset(path="/path/to/asset2", name="asset2")
    res = concrete_resource_pybind.CppConcreteResource([asset1, asset2])

    # CppConreteResource::Load is expected to return an error.
    with self.assertRaisesRegex(_STATUS_NOT_OK, "CppConcreteResource"):
      res.load()

    self.assertIsNotNone(res.name)
    self.assertEqual({a.name for a in res.assets}, {"asset1", "asset2"})
    self.assertEqual(
        {a.path for a in res.assets}, {"/path/to/asset1", "/path/to/asset2"}
    )
    self.assertEqual(res.class_name, "CppConcreteResource")

  def test_to_proto(self):
    asset1 = resource_manager.Asset(path="/path/to/asset1", name="asset1")
    asset2 = resource_manager.Asset(path="/path/to/asset2", name="asset2")
    asset3 = resource_manager.Asset(path="/path/to/asset3", name="asset3")
    res1 = concrete_resource_pybind.CppConcreteResource([asset1, asset2])
    res2 = concrete_resource_pybind.OkConcreteResource([asset2])
    res3 = another_concrete_resource_pybind.AnotherConcreteResource([asset3])
    res4 = another_concrete_resource_pybind.AnotherConcreteResource([])
    rm = resource_manager.ResourceManager([res1, res2])
    rm.add_resource(res3)
    rm.add_resource(res4)

    proto = rm.to_proto()
    expected = text_format.Parse(
        _TEST_PROTO_MESSAGE.format(
            id1=res1.id,
            res1=res1.name,
            id2=res2.id,
            res2=res2.name,
            id3=res3.id,
            res3=res3.name,
            id4=res4.id,
            res4=res4.name,
        ),
        resource_manager_pb2.ResourceManagerProto(),
    )
    test_utils.assert_proto2_equal(
        self, proto, expected, ignore_repeated_fields_order=True
    )

  def test_from_proto(self):
    proto = text_format.Parse(
        _TEST_PROTO_MESSAGE.format(
            id1=1,
            res1="res1",
            id2=2,
            res2="res2",
            id3=3,
            res3="res3",
            id4=4,
            res4="res4",
        ),
        resource_manager_pb2.ResourceManagerProto(),
    )
    rm = resource_manager.ResourceManager.from_proto(proto)

    # Verify resources.
    resources = rm.get_all_resources()
    resources.sort(key=lambda x: x.name)
    res1, res2, res3, res4 = resources
    names = {r.name for r in resources}
    class_names = {r.class_name for r in resources}
    self.assertEqual(len(resources), 4)
    self.assertEqual(names, {"res1", "res2", "res3", "res4"})
    self.assertEqual(
        class_names,
        {
            "CppConcreteResource",
            "OkConcreteResource",
            "AnotherConcreteResource",
        },
    )
    self.assertEqual({a.name for a in res1.assets}, {"asset1", "asset2"})
    self.assertEqual(
        {a.path for a in res1.assets}, {"/path/to/asset1", "/path/to/asset2"}
    )
    self.assertEqual({a.name for a in res2.assets}, {"asset2"})
    self.assertEqual({a.path for a in res2.assets}, {"/path/to/asset2"})
    self.assertEqual({a.name for a in res3.assets}, {"asset3"})
    self.assertEqual({a.path for a in res3.assets}, {"/path/to/asset3"})
    self.assertEqual(res4.assets, [])
    for res in resources:
      self.assertIsInstance(res, resource_manager.CppResource)
    with self.assertRaisesRegex(_STATUS_NOT_OK, "CppConcreteResource"):
      res1.load()
    res2.load()  # OkConcreteResource should not raise error.
    with self.assertRaisesRegex(_STATUS_NOT_OK, "AnotherConcreteResource"):
      res3.load()
    with self.assertRaisesRegex(_STATUS_NOT_OK, "AnotherConcreteResource"):
      res4.load()

    # Verify get_all_assets api.
    assets = rm.get_all_assets()
    asset_names = {a.name for a in assets}
    asset_paths = {a.path for a in assets}
    self.assertEqual(asset_names, {"asset1", "asset2", "asset3"})
    self.assertEqual(
        asset_paths, {"/path/to/asset1", "/path/to/asset2", "/path/to/asset3"}
    )


if __name__ == "__main__":
  absltest.main()
