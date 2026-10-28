from types import ModuleType

import pytest

from Qt._lazy import export_module


def test_exports_are_lazy_and_preserve_shim_aliases():
    calls = []
    binding = ModuleType('binding')
    values = {'Widget': object(), 'Relocated': object()}
    binding.__dir__ = lambda: list(values)
    binding.__getattr__ = lambda name: calls.append(name) or values[name]
    shim = ModuleType('shim')
    relocated = shim.Relocated = object()
    export_module(vars(shim), binding)
    assert not calls
    assert 'Widget' in dir(shim)
    assert shim.Relocated is relocated
    assert shim.Widget is values['Widget']
    assert shim.Widget is values['Widget']
    assert calls == ['Widget']
    assert set(shim.__all__) == {'Widget', 'Relocated'}
    with pytest.raises(AttributeError, match='Missing'):
        shim.Missing


def test_wildcard_import_keeps_public_names(monkeypatch):
    import sys
    binding = ModuleType('binding')
    binding.A = object()
    binding._private = object()
    shim = ModuleType('lazy_shim_example')
    shim.Alias = object()
    export_module(vars(shim), binding)
    monkeypatch.setitem(sys.modules, shim.__name__, shim)
    target = {}
    exec('from lazy_shim_example import *', target)
    assert target['A'] is binding.A
    assert target['Alias'] is shim.Alias
    assert '_private' not in target


def test_preparation_waits_for_access_and_is_cached():
    binding = ModuleType('binding')
    binding.Widget = object()
    shim = ModuleType('shim')
    calls = []
    prepared = object()
    def prepare(value):
        calls.append(value)
        return prepared
    export_module(vars(shim), binding, prepare={'Widget': prepare})
    assert 'Widget' in dir(shim) and 'Widget' in shim.__all__
    assert not calls
    assert shim.Widget is prepared
    assert shim.Widget is prepared
    assert calls == [binding.Widget]


def test_failed_preparation_can_retry():
    binding = ModuleType('binding')
    binding.Widget = object()
    shim = ModuleType('shim')
    calls = []
    def prepare(value):
        calls.append(value)
        if len(calls) == 1:
            raise RuntimeError('not ready')
        return value
    export_module(vars(shim), binding, prepare={'Widget': prepare})
    with pytest.raises(RuntimeError, match='not ready'):
        shim.Widget
    assert 'Widget' not in vars(shim)
    assert shim.Widget is binding.Widget


def test_file_dialog_preparation_preserves_static_calls():
    from Qt.QtWidgets import _prepare_pyside_file_dialog
    calls = []
    class Dialog:
        pass
    for name in ('getExistingDirectory', 'getOpenFileName', 'getOpenFileNames', 'getSaveFileName'):
        def original(*args, _name=name, **kwargs):
            calls.append((_name, args, kwargs))
            return 'result'
        setattr(Dialog, name, staticmethod(original))
    assert _prepare_pyside_file_dialog(Dialog) is Dialog
    for name in ('getExistingDirectory', 'getOpenFileName', 'getOpenFileNames', 'getSaveFileName'):
        assert getattr(Dialog, name)('parent', directory='/tmp') == 'result'
        assert calls[-1] == (name, ('parent',), {'dir': '/tmp'})
        assert getattr(Dialog(), name)(dir='/tmp') == 'result'
        assert calls[-1] == (name, (), {'dir': '/tmp'})
