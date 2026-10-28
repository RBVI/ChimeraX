"""Settings factories read current values on first display and run only once."""
from types import SimpleNamespace

from chimerax.ui.gui import MainWindow


def test_pending_factories_read_current_settings_and_preserve_registration_order():
    window = SimpleNamespace(_settings_ui_widget=None, _accumulated_settings_options=[])
    state = {'value': 'initial'}
    creations = []
    def create():
        option = SimpleNamespace(value=state['value'])
        creations.append(option)
        return option
    existing = object()
    MainWindow.add_settings_option(window, 'Window', create, factory=True)
    MainWindow.add_settings_option(window, 'Startup', existing)
    assert creations == []
    state['value'] = 'changed before opening Settings'
    options = []
    panel = SimpleNamespace(add_option=lambda category, option: options.append((category, option)))
    MainWindow._add_accumulated_settings_options(window, panel)
    assert len(creations) == 1
    assert options == [('Window', creations[0]), ('Startup', existing)]
    assert creations[0].value == state['value']
    MainWindow._add_accumulated_settings_options(window, panel)
    assert len(options) == 2
    assert len(creations) == 1


def test_options_registered_after_settings_opens_are_created_immediately():
    calls = []
    panel = SimpleNamespace(add_option=lambda *args: calls.append(args))
    window = SimpleNamespace(_settings_ui_widget=object(),
        _core_settings_panel=SimpleNamespace(options_widget=panel))
    option = object()
    MainWindow.add_settings_option(window, 'Window', lambda: option, factory=True)
    MainWindow.add_settings_option(window, 'Startup', option)
    assert calls == [('Window', option), ('Startup', option)]


def test_factory_reads_default_from_settings_at_construction():
    from Qt.QtWidgets import QApplication
    from chimerax.ui.gui import ToolSideOption
    from chimerax.core.settings import Settings
    import pytest

    app = QApplication.instance() or QApplication([])
    if not isinstance(app, QApplication):
        pytest.skip('Requires a GUI application')
    class ToolSettings(Settings):
        EXPLICIT_SAVE = {'side': 'left'}
    settings = ToolSettings(None, 'deferred-settings-test')
    window = SimpleNamespace(_settings_ui_widget=None, _accumulated_settings_options=[])
    MainWindow.add_settings_option(window, 'Window',
        lambda: ToolSideOption('Default tool side', settings.side, None), factory=True)
    settings.side = 'right'
    options = []
    panel = SimpleNamespace(add_option=lambda category, option: options.append(option))
    MainWindow._add_accumulated_settings_options(window, panel)
    try:
        assert options[0].value == 'right'
    finally:
        options[0].widget.deleteLater()
