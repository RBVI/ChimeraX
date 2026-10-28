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


