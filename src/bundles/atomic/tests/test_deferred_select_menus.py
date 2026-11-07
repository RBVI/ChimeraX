"""Atomic menu extensions and standalone element pickers retain their behavior."""
from types import SimpleNamespace

import pytest


@pytest.fixture
def window(monkeypatch):
    from Qt.QtWidgets import QApplication, QMainWindow
    from chimerax.ui.gui import MainWindow
    from chimerax import atomic
    app = QApplication.instance() or QApplication([])
    class Window(QMainWindow):
        add_select_submenu = MainWindow.add_select_submenu
        add_menu_selector = MainWindow.add_menu_selector
        _get_target_menu = MainWindow._get_target_menu
        _get_menu_action = MainWindow._get_menu_action
        add_menu_entry = MainWindow.add_menu_entry
        def select_by_mode(self, text):
            self.selected.append(text)
    window = Window()
    window.selected = []
    window.menuBar().addMenu('Select').addAction('Sequence...')
    window.session = SimpleNamespace()
    monkeypatch.setattr(atomic, 'get_triggers', lambda: SimpleNamespace(add_handler=lambda *args: None))
    yield window
    window.close()
    window.deleteLater()


def test_menu_registration_defers_type_data_and_first_show_populates_once(window, monkeypatch):
    from chimerax import atomic
    from chimerax.atomic.selectors import add_select_menu_items
    from chimerax.ui.gui import _ensure_menu_contents
    accesses = []
    class Elements:
        @property
        def names(self):
            accesses.append('elements')
            return ['H', 'He', 'C', 'N', 'O', 'S', 'P']
    class Atoms:
        @property
        def idatm_info_map(self):
            accesses.append('types')
            return {'C3': None, 'N3': None}
    monkeypatch.setattr(atomic, 'Element', Elements())
    monkeypatch.setattr(atomic, 'Atom', Atoms())
    session = SimpleNamespace(ui=SimpleNamespace(main_window=window))
    add_select_menu_items(session)
    assert not accesses
    # Extension lookup materializes IDATM before looking up an insertion point.
    types = window.add_select_submenu(['Chemistry'], 'IDATM Type')
    assert accesses == ['types']
    assert [action.text() for action in types.actions()] == ['C3', 'N3']
    types.actions()[0].trigger()
    assert window.selected == ['C3']
    _ensure_menu_contents(types)
    assert accesses == ['types']
    elements = window.add_select_submenu(['Chemistry', 'Element'], 'Other')
    assert accesses == ['types', 'elements']
    entries = [action.text() for action in elements.actions()]
    assert entries and all(action.menu() for action in elements.actions())
    _ensure_menu_contents(elements)
    assert [action.text() for action in elements.actions()] == entries


def test_standalone_element_picker_is_eager_and_complete(window):
    from chimerax.atomic import Element
    from chimerax.atomic.selectors import make_elements_menu
    picker = make_elements_menu(window)
    common = [action.text() for action in picker.actions()[:6]]
    assert common == ['C', 'H', 'N', 'O', 'P', 'S']
    other = picker.actions()[-1].menu()
    names = [item.text() for section in other.actions() for item in section.menu().actions()]
    assert names == sorted(name for name in Element.names if len(name) < 3)
