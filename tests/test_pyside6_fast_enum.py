"""prereqs/pips/pyside6_fast_enum.py builds the same enum classes as the
functional API for the calls libshiboken makes, and leaves others alone."""

import enum
import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "prereqs/pips/pyside6_fast_enum.py"


@pytest.fixture
def fast_enum(monkeypatch):
    monkeypatch.setattr(enum.EnumType, "_create_", enum.EnumType._create_)
    spec = importlib.util.spec_from_file_location("pyside6_fast_enum", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reference(module, base, name, items):
    saved = enum.EnumType._create_
    enum.EnumType._create_ = module._create
    try:
        return base(name, items, boundary=enum.KEEP)
    finally:
        enum.EnumType._create_ = saved


@pytest.mark.parametrize("base", [enum.Enum, enum.IntEnum, enum.Flag, enum.IntFlag])
def test_same_classes_as_the_functional_api(fast_enum, base):
    # Flags with combined values and an alias, as Qt enums have.
    items = [("A", 1), ("B", 2), ("C", 4), ("AB", 3), ("Alias", 1), ("All", 7)]
    fast = base("Sample", list(items), boundary=enum.KEEP)
    assert getattr(enum.EnumType._create_, "_chimerax_fast_enum", False)
    enum._test_simple_enum(_reference(fast_enum, base, "Sample", list(items)), fast)
    assert fast(1) is fast.A and fast.Alias is fast.A


def test_other_calls_use_the_functional_api(fast_enum):
    calls = []
    original = fast_enum._create

    def record(*args, **kw):
        calls.append(args[1])
        return original(*args, **kw)
    fast_enum._create = record
    enum.IntEnum("Default", [("A", 1)])                          # no boundary
    enum.IntEnum("Names", "A B", boundary=enum.KEEP)             # string of names
    with pytest.raises(TypeError):                               # duplicate names still fail
        enum.IntEnum("Dup", [("A", 1), ("A", 2)], boundary=enum.KEEP)
    assert calls == ["Default", "Names", "Dup"]
