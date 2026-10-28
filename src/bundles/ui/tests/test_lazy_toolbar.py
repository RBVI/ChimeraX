import pytest

from Qt.QtWidgets import QApplication
from chimerax.ui.widgets.tabbedtoolbar import TabbedToolbar


@pytest.fixture
def toolbar():
    app = QApplication.instance() or QApplication([])
    if not isinstance(app, QApplication):
        pytest.skip('Toolbar widgets require a GUI application')
    widget = TabbedToolbar(lazy_tabs=True)
    yield widget
    widget.close()
    widget.deleteLater()


def test_hidden_buttons_preserve_state_and_callbacks_until_selection(toolbar):
    calls = []
    toolbar.add_button('Home', 'Main', 'Home', None)
    toolbar.add_button('Later', 'Actions', 'Run', lambda: calls.append('run'))
    section = toolbar._buttons['Later']['Actions']
    assert toolbar._buttons['Home']['Main'].createdWidgets()
    assert not section.createdWidgets()
    toolbar.set_enabled(False, 'Later', 'Actions', 'Run')
    toolbar.add_button_highlight('Later', 'Actions', 'Run')
    toolbar.set_show_button_titles(False)
    toolbar.show_tab('Later')
    parent = section.createdWidgets()[0]
    action = section.get_qt_button_action(parent, 'Run')
    assert not action.isEnabled()
    assert not action.icon().isNull()
    assert not section.show_button_titles
    toolbar.set_enabled(True, 'Later', 'Actions', 'Run')
    action.trigger()
    assert calls == ['run']
    toolbar.show_tab('Home')
    toolbar.show_tab('Later')
    assert section.createdWidgets() == [parent]


def test_hidden_group_selection_is_applied_when_widgets_are_created(toolbar):
    toolbar.add_button('Home', 'Main', 'Home', None)
    toolbar.add_button('Later', 'Actions', 'First', None, group='choice')
    toolbar.add_button('Later', 'Actions', 'Second', None, group='choice')
    section = toolbar._buttons['Later']['Actions']
    toolbar.show_group_button('Later', 'Actions', 'Second')
    assert not section.createdWidgets()
    toolbar.show_tab('Later')
    parent = section.createdWidgets()[0]
    assert section._groups[parent]['choice'].defaultAction() is section.get_qt_button_action(parent, 'Second')


def test_clearing_hidden_tab_discards_old_buttons(toolbar):
    toolbar.add_button('Home', 'Main', 'Home', None)
    toolbar.add_button('Later', 'Actions', 'Old', None)
    toolbar.clear_tab('Later')
    toolbar.add_button('Later', 'Replacement', 'New', None)
    toolbar.show_tab('Later')
    assert set(toolbar._buttons['Later']) == {'__toolbar__', 'Replacement'}
    assert toolbar._buttons['Later']['Replacement'].createdWidgets()
    toolbar.clear_all()
    assert toolbar.count() == 0
    assert not toolbar._unpopulated_tabs
    toolbar.add_button('New Home', 'Main', 'New', None)
    assert toolbar._buttons['New Home']['Main'].createdWidgets()


def test_icon_factory_waits_for_tab_and_survives_layout_rebuild(toolbar):
    from Qt.QtGui import QIcon, QPixmap
    calls = []
    pixmap = QPixmap(16, 16)
    pixmap.fill()
    icon = QIcon(pixmap)
    def load_icon():
        calls.append(True)
        return icon
    toolbar.add_button('Home', 'Main', 'Home', None)
    toolbar.add_button('Later', 'Actions', 'Run', None, icon=load_icon)
    assert calls == []
    toolbar.set_enabled(False, 'Later', 'Actions', 'Run')
    toolbar.show_tab('Later')
    section = toolbar._buttons['Later']['Actions']
    parent = section.createdWidgets()[0]
    action = section.get_qt_button_action(parent, 'Run')
    assert action.icon().cacheKey() == icon.cacheKey()
    assert not action.isEnabled()
    toolbar.set_show_button_titles(False)
    toolbar.show_tab('Home')
    toolbar.show_tab('Later')
    assert calls == [True]


def test_highlight_waits_for_hidden_tab_and_resolves_icon_once(toolbar):
    calls = []
    def no_icon():
        calls.append(True)
        return None
    toolbar.add_button('Home', 'Main', 'Home', None)
    toolbar.add_button('Later', 'Modes', 'Move', None, icon=no_icon)
    toolbar.add_button_highlight('Later', 'Modes', 'Move')
    section = toolbar._buttons['Later']['Modes']
    assert not section.createdWidgets()
    assert calls == []
    toolbar.show_tab('Later')
    parent = section.createdWidgets()[0]
    assert not section.get_qt_button_action(parent, 'Move').icon().isNull()
    assert calls == [True]
    toolbar.remove_button_highlight('Later', 'Modes', 'Move')
    toolbar.set_show_button_titles(False)
    assert section._buttons[0].icon is None
    assert calls == [True]


def test_clearing_unvisited_tab_does_not_load_icons(toolbar):
    def unexpected():
        raise AssertionError('Discarded buttons must not load their icons')
    toolbar.add_button('Home', 'Main', 'Home', None)
    toolbar.add_button('Later', 'Actions', 'Old', None, icon=unexpected)
    toolbar.clear_tab('Later')
    toolbar.add_button('Later', 'Actions', 'New', None)
    toolbar.show_tab('Later')


def test_highlight_on_visible_tab_is_drawn_immediately(toolbar):
    from Qt.QtGui import QColor
    toolbar.add_button('Home', 'Modes', 'Move', None)
    section = toolbar._buttons['Home']['Modes']
    parent = section.createdWidgets()[0]
    assert section.get_qt_button_action(parent, 'Move').icon().isNull()
    toolbar.add_button_highlight('Home', 'Modes', 'Move')
    parent = section.createdWidgets()[0]
    icon = section.get_qt_button_action(parent, 'Move').icon()
    assert not icon.isNull()
    toolbar.remove_button_highlight('Home', 'Modes', 'Move')
    assert section.get_qt_button_action(parent, 'Move').icon().isNull()
    section.highlight_color = QColor('red')
    toolbar.add_button_highlight('Home', 'Modes', 'Move')
    image = section.get_qt_button_action(parent, 'Move').icon().pixmap(8, 8).toImage()
    assert image.pixelColor(4, 4).red() == 255


def test_changing_highlight_color_redraws_only_highlighted_buttons(toolbar):
    from Qt.QtGui import QColor
    toolbar.add_button('Home', 'Main', 'Home', None)
    toolbar.add_button('Later', 'Modes', 'Move', None)
    toolbar.add_button('Later', 'Modes', 'Turn', None)
    toolbar.add_button_highlight('Later', 'Modes', 'Move')
    section = toolbar._buttons['Later']['Modes']
    section.set_highlight_color(QColor('red'))
    move, turn = section._buttons
    assert move.highlighted and not turn.highlighted
    assert move.highlight_icon.pixmap(8, 8).toImage().pixelColor(4, 4).red() == 255
