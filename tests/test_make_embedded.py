import importlib.util
import pathlib
from types import SimpleNamespace


def _load_make_embedded_module():
    path = pathlib.Path(__file__).resolve().parents[1] / "docs" / "make_embedded.py"
    spec = importlib.util.spec_from_file_location("make_embedded", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_find_license_file_ignores_oserror_from_top_level_import(monkeypatch):
    make_embedded = _load_make_embedded_module()
    pkg = SimpleNamespace(files=[], locate_file=lambda _path: None)

    monkeypatch.setattr(
        make_embedded.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(OSError("cannot load shared object")),
    )

    assert make_embedded.find_license_file(pkg) is None
