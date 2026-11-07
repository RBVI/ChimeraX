"""Build unused menus on demand while retaining bundle extensions and tool state."""
from types import SimpleNamespace

import pytest


@pytest.fixture
def window():
    from Qt.QtWidgets import QApplication, QMenuBar, QWidget
    from chimerax.ui.gui import MainWindow

    app = QApplication.instance() or QApplication([])
    if not isinstance(app, QApplication):
        pytest.skip('Requires a GUI application')

    class MenuWindow(QWidget):
        update_tools_menu = MainWindow.update_tools_menu
        _populate_tools_menu = MainWindow._populate_tools_menu
        _update_running_tools = MainWindow._update_running_tools
        _set_tool_checkbuttons = MainWindow._set_tool_checkbuttons
        _get_target_menu = MainWindow._get_target_menu
        _get_menu_action = MainWindow._get_menu_action
        add_menu_entry = MainWindow.add_menu_entry

        def menuBar(self):
            return self.bar

    widget = MenuWindow()
    widget.bar = QMenuBar(widget)
    widget.tools_menu = widget.bar.addMenu('Tools')
    widget._toolbar_visibility = {}
    widget.tool_instance_to_windows = {}
    yield widget
    widget.close()
    widget.deleteLater()


def test_menu_builds_once_before_display_and_allows_reentrant_access(window):
    from chimerax.ui.gui import _defer_menu_contents, _ensure_menu_contents

    menu = window.bar.addMenu('Actions')
    calls = []

    def populate():
        calls.append('populate')
        _ensure_menu_contents(menu)
        menu.addAction('Run').triggered.connect(lambda: calls.append('run'))

    _defer_menu_contents(menu, populate)
    assert calls == []
    # Nonempty even before population, so native menus can be opened.
    assert menu.actions()
    menu.aboutToShow.emit()
    assert [action.text() for action in menu.actions()] == ['Run']
    menu.aboutToShow.emit()
    _ensure_menu_contents(menu)
    assert calls == ['populate']
    menu.actions()[0].trigger()
    assert calls == ['populate', 'run']


def test_extension_finds_builtin_submenu_and_insertion_point_before_first_show(window):
    from chimerax.ui.gui import _defer_menu_contents

    menu = window.bar.addMenu('Actions')
    menu.setObjectName('Actions')
    calls = []

    def populate():
        color = menu.addMenu('Color')
        color.addAction('Red')
        calls.append('populate')

    _defer_menu_contents(menu, populate)
    extension = window.add_menu_entry(['Actions', 'Color'], 'Custom',
        lambda: calls.append('custom'), insertion_point='Red')
    color = menu.actions()[0].menu()
    assert [action.text() for action in color.actions()] == ['Custom', 'Red']
    menu.aboutToShow.emit()
    assert len(menu.actions()) == 1
    extension.trigger()
    assert calls == ['populate', 'custom']


def test_extension_to_top_level_menu_follows_builtin_items(window):
    from chimerax.ui.gui import _defer_menu_contents

    menu = window.bar.addMenu('Actions')
    menu.setObjectName('Actions')
    _defer_menu_contents(menu, lambda: menu.addAction('Focus'))
    window.add_menu_entry(['Actions'], 'Extension', lambda: None)
    assert [action.text() for action in menu.actions()] == ['Focus', 'Extension']


def test_real_actions_menu_retains_color_icons_and_commands(window):
    from chimerax.ui.gui import MainWindow, _defer_menu_contents

    commands = []
    window.session = SimpleNamespace()
    window._run_surf_command = lambda command, **kw: commands.append(command)
    window._get_label_text_arg = lambda: ''
    window.color_by_editor = lambda: None
    window.show_set_label_height_dialog = lambda: None
    menu = window.bar.addMenu('Actions')
    menu.setObjectName('Actions')
    _defer_menu_contents(menu, lambda: MainWindow._populate_actions_menu(window, menu))
    assert len(menu.actions()) == 1
    assert menu.actions()[0].icon().isNull()
    window.add_menu_entry(['Actions', 'Color'], 'Extension', lambda: None,
        insertion_point='Red')
    color = window._get_target_menu(window.bar, ['Actions', 'Color'])
    red = color.actions()[1]
    assert red.text() == 'Red'
    assert not red.icon().isNull()
    red.trigger()
    assert commands == ['color %s red']
    contents = menu.actions()
    menu.aboutToShow.emit()
    assert menu.actions() == contents


class Tool:
    def __init__(self, name, category):
        self.name = name
        self.categories = [category]
        self.synopsis = name + ' help'


def tool_session(tools, active=()):
    bundles = [SimpleNamespace(tools=tools)]
    return SimpleNamespace(logger=None,
        toolshed=SimpleNamespace(bundle_info=lambda logger: bundles),
        tools=SimpleNamespace(list=lambda: [SimpleNamespace(display_name=name) for name in active]))


def tool_actions(menu):
    return [action for action in menu.actions()
            if not action.isSeparator() and action.text() != 'More Tools...']


def test_tools_metadata_available_before_actions_and_reload_replaces_pending_menu(window):
    old = Tool('Old Tool', 'General')
    new = Tool('New Tool', 'General')
    window.update_tools_menu(tool_session([old]))
    assert window._tools_cache == {old}
    assert window._checkbutton_tools == {}
    window.update_tools_menu(tool_session([new]))
    assert window._tools_cache == {new}
    # Emitting the real signal must build only the latest catalog, once.
    window.tools_menu.aboutToShow.emit()
    assert [action.text() for action in tool_actions(window.tools_menu)] == ['New Tool']
    window.tools_menu.aboutToShow.emit()
    assert [action.text() for action in tool_actions(window.tools_menu)] == ['New Tool']


def test_toolbar_visibility_before_first_show_is_retained_and_callbacks_work(window, monkeypatch):
    from chimerax.core import commands

    calls = []
    monkeypatch.setattr(commands, 'run', lambda session, command: calls.append(command))
    tool = Tool('Example Toolbar', 'Toolbar')
    session = tool_session([tool], active=[tool.name])
    window.update_tools_menu(session)
    toolbar = SimpleNamespace(windowTitle=lambda: tool.name)
    window._set_tool_checkbuttons(toolbar, False)
    assert window._checkbutton_tools == {}
    # The running-tools callback can also be called directly, before show.
    window._update_running_tools()
    action = window._checkbutton_tools[tool.name]
    assert not action.isChecked()
    action.trigger()
    assert calls == ['ui tool show "Example Toolbar"']
    window._set_tool_checkbuttons(toolbar, False)
    assert not action.isChecked()
    window.update_tools_menu(session)
    window.tools_menu.aboutToShow.emit()
    assert not window._checkbutton_tools[tool.name].isChecked()


def test_tools_reload_after_population_discards_previous_actions(window):
    first = Tool('First', 'General')
    second = Tool('Second', 'General')
    window.update_tools_menu(tool_session([first]))
    window.tools_menu.aboutToShow.emit()
    window.update_tools_menu(tool_session([second]))
    window.tools_menu.aboutToShow.emit()
    assert [action.text() for action in tool_actions(window.tools_menu)] == ['Second']


def test_multiple_tool_categories_and_launch_command(window, monkeypatch):
    from chimerax.core import commands

    calls = []
    monkeypatch.setattr(commands, 'run', lambda session, command: calls.append(command))
    window.update_tools_menu(tool_session([
        Tool('Z Tool', 'General'), Tool('A Tool', 'General'), Tool('Toolbar', 'Toolbar')]))
    window.tools_menu.aboutToShow.emit()
    categories = [action.menu() for action in window.tools_menu.actions() if action.menu()]
    assert [menu.title() for menu in categories] == ['General', 'Toolbar']
    assert [action.text() for action in categories[0].actions()] == ['A Tool', 'Z Tool']
    categories[0].actions()[0].trigger()
    assert calls == ['ui tool show "A Tool"']
