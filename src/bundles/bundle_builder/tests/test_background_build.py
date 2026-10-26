import importlib
import os

import pytest


@pytest.fixture(params=["bundle_builder", "bundle_builder_toml"])
def builder(request):
    return importlib.import_module("chimerax.bundle_builder." + request.param)


def test_each_extension_has_its_own_object_folder(builder, monkeypatch):
    class Ext:
        name = "chimerax.example._ext"

    seen = []
    monkeypatch.setattr(
        builder._build_ext,
        "build_extension",
        lambda self, ext: seen.append(self.build_temp),
    )
    command = object.__new__(builder._BuildExt)
    command.build_temp = os.path.join("build", "temp")
    command.build_extension(Ext())
    assert seen == [os.path.join("build", "temp", Ext.name)]
    # The command the build_ext -j threads share is unchanged
    assert command.build_temp == os.path.join("build", "temp")
