"""Run against each supported PySide release when updating the binding."""
from enum import EnumType
from types import SimpleNamespace

import pytest

PySide6 = pytest.importorskip('PySide6')
from PySide6 import QtCore, QtGui, QtWidgets
from Qt.promote_enums import native_enum_shortcuts, promote_enums_pyside


@pytest.mark.parametrize('module', [QtCore, QtGui, QtWidgets])
def test_all_promoted_aliases_keep_the_same_value(module):
    expected = []
    wrappertype = type(QtWidgets.QWidget)
    for name in dir(module):
        if not name.startswith('Q'):
            continue
        klass = getattr(module, name)
        if not isinstance(klass, wrappertype):
            continue
        aliases = {}
        for attribute in dir(klass):
            if attribute[0].isupper():
                value = getattr(klass, attribute)
                if isinstance(value, EnumType):
                    aliases.update(value.__members__)
        expected.extend((klass, name, value) for name, value in aliases.items())
    promote_enums_pyside(module)
    for klass, name, value in expected:
        assert getattr(klass, name) is value, (klass, name)


@pytest.mark.parametrize('mode', [0, 0x11, 0x21, 0x0f, None])
def test_custom_enum_modes_retain_explicit_promotion(monkeypatch, mode):
    import sys
    monkeypatch.setattr(sys, 'pyside6_option_python_enum', mode, raising=False)
    assert not native_enum_shortcuts(QtWidgets)


def test_other_modules_and_old_bindings_retain_explicit_promotion(monkeypatch):
    assert not native_enum_shortcuts(SimpleNamespace(__name__='PySide6.QtNetwork'))
    monkeypatch.setattr(PySide6, '__version_info__', (6, 4, 0))
    assert not native_enum_shortcuts(QtWidgets)


def test_exported_widget_instances_keep_unscoped_enums():
    from Qt.QtWidgets import QApplication, QTableView
    app = QApplication.instance() or QApplication([])
    table = QTableView()
    try:
        assert table.AdjustToContents is QTableView.SizeAdjustPolicy.AdjustToContents
        assert table.SelectRows is QTableView.SelectionBehavior.SelectRows
    finally:
        table.deleteLater()


def test_lazy_preparation_populates_actual_class_attributes():
    from Qt.promote_enums import prepare_pyside_class
    from PySide6.QtCore import QEvent
    assert prepare_pyside_class(QEvent) is QEvent
    assert vars(QEvent)['MouseButtonPress'] is QEvent.Type.MouseButtonPress
