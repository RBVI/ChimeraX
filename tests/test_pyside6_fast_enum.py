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
    # Qt may have installed its copy during collection. Exercise this source
    # module, and restore the installed wrapper at fixture teardown.
    monkeypatch.setattr(enum.EnumType, '_create_', module._create_)
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


@pytest.mark.parametrize('base', [enum.Enum, enum.IntEnum, enum.Flag, enum.IntFlag])
@pytest.mark.parametrize('values', [
    [('Zero', 0), ('High', 32), ('Low', 1), ('Combined', 33), ('Alias', 1)],
    [('Negative', -1), ('Zero', 0), ('One', 1)],
    [],
])
def test_integer_metadata_and_aliases_match_the_standard_library(fast_enum, base, values):
    reference = _reference(fast_enum, base, 'Sample', values)
    result = base('Sample', list(values), boundary=enum.KEEP)
    enum._test_simple_enum(reference, result)
    assert [member.value for member in reference] == [member.value for member in result]
    if values and issubclass(base, enum.Flag):
        for value in (0, 1, 2, 7, 32, 63, -1):
            assert result(value).value == reference(value).value
            assert (~result(value)).value == (~reference(value)).value


def test_custom_names_and_values_keep_standard_handling(fast_enum):
    for values in ([('name', 1), ('value', 2)], [('One', 'text')], [('One', True)], [('Bad-name', 1)]):
        reference = _reference(fast_enum, enum.Enum, 'Sample', values)
        result = enum.Enum('Sample', list(values), boundary=enum.KEEP)
        enum._test_simple_enum(reference, result)


def test_all_installed_pyside_integer_enums_match_the_reference(fast_enum):
    import importlib
    pyside = pytest.importorskip('PySide6')
    modules = sorted({path.name.split('.')[0]
        for directory in pyside.__path__ for path in Path(directory).glob('Qt*')
        if path.suffix in ('.so', '.pyd')})
    checked = set()
    for module_name in modules:
        module = importlib.import_module('PySide6.' + module_name)
        for name in dir(module):
            qt_class = getattr(module, name)
            if not isinstance(qt_class, type):
                continue
            for actual in tuple(vars(qt_class).values()):
                if not isinstance(actual, enum.EnumType) or actual in checked:
                    continue
                checked.add(actual)
                values = [(name, member.value) for name, member in actual.__members__.items()]
                base = next(base for base in (enum.IntFlag, enum.Flag, enum.IntEnum, enum.Enum)
                            if issubclass(actual, base))
                reference = _reference(fast_enum, base, actual.__name__, values)
                result = base(actual.__name__, list(values), boundary=enum.KEEP)
                enum._test_simple_enum(reference, result)
    assert len(checked) > 100, 'Expected the installed Qt bindings to expose their enums'
