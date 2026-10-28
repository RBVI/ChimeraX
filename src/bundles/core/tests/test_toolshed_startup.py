"""Startup costs of the Toolshed's available-bundle catalog."""

import os
from types import SimpleNamespace


def test_catalog_snapshot_round_trip_and_invalidation(tmp_path):
    from chimerax.core.toolshed import available
    path = tmp_path / "available.json"
    path.write_text("[]")
    data = [["format_version", 3], {"bundle_name": "ChimeraX_Example", "bundle": {}}]
    stat = os.stat(path)
    assert available._read_snapshot(str(path), stat) is None
    available._write_snapshot(str(path), stat, data)
    assert available._read_snapshot(str(path), stat) == data
    # A changed catalog makes the snapshot stale.
    path.write_text("[1]")
    assert available._read_snapshot(str(path), os.stat(path)) is None
    # A damaged snapshot is ignored.
    (tmp_path / "available.json.marshal").write_bytes(b"not marshal data")
    assert available._read_snapshot(str(path), os.stat(path)) is None


def test_managers_can_skip_uninstalled_providers():
    from chimerax.core.toolshed import Toolshed, ProviderManager

    def bundle(name, installed):
        return SimpleNamespace(name=name, installed=installed, providers={"mgr/p": {}},
                               _providers_for_manager=lambda m: [("p", {})] if m == "mgr" else [])

    class Manager(ProviderManager):
        def __init__(self, uses_uninstalled):
            self.name = "mgr"
            self.uses_uninstalled_providers = uses_uninstalled
            self.added = []
            self.ended = False

        def add_provider(self, bundle_info, name, **kw):
            self.added.append(bundle_info.name)

        def end_providers(self):
            self.ended = True

    ts = Toolshed.__new__(Toolshed)
    ts._installed_bundle_info = [bundle("installed", True)]
    ts._available_bundle_info = [bundle("available", False)]
    ts._manager_instances = {}
    every = Manager(True)
    ts._init_single_manager(every)
    assert every.added == ["installed", "available"] and every.ended
    installed_only = Manager(False)
    ts._init_single_manager(installed_only)
    assert installed_only.added == ["installed"] and installed_only.ended
