"""Regression coverage for menu registration without GUI class imports."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest


@pytest.fixture
def dialog_module():
    path = Path(__file__).parents[1] / 'src' / 'dialog.py'
    spec = importlib.util.spec_from_file_location('chimerax.open_command._dialog_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_loading_menu_helpers_does_not_define_dialogs(dialog_module):
    for name in ('OpenDialog', 'OpenFolderDialog', 'FetchDialog'):
        assert name not in vars(dialog_module)
        assert name in dir(dialog_module)
        assert name in dialog_module.__all__
    with pytest.raises(AttributeError, match='missing'):
        dialog_module.missing


def test_menu_registration_keeps_dialogs_deferred(dialog_module):
    entries = []
    window = SimpleNamespace(add_menu_entry=lambda *args, **kw: entries.append((args, kw)))
    session = SimpleNamespace(open_command=SimpleNamespace(open_data_formats=[]),
                              ui=SimpleNamespace(main_window=window))
    dialog_module.create_menu_entry(session)
    assert [e[0][1] for e in entries] == ['&Fetch By ID...', '&Open...']
    assert all(name not in vars(dialog_module) for name in dialog_module._DIALOG_CLASSES)


def test_lazy_classes_preserve_identity_and_inheritance(dialog_module):
    from chimerax.open_command import _dialog_gui
    from Qt.QtWidgets import QFileDialog, QDialog
    for name in dialog_module._DIALOG_CLASSES:
        assert getattr(dialog_module, name) is getattr(_dialog_gui, name)
        assert getattr(dialog_module, name) is vars(dialog_module)[name]
    assert issubclass(dialog_module.OpenDialog, QFileDialog)
    assert issubclass(dialog_module.OpenFolderDialog, dialog_module.OpenDialog)
    assert issubclass(dialog_module.FetchDialog, QDialog)
