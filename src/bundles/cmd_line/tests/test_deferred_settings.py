"""Command-line preference editors are only needed when Settings opens."""
from types import SimpleNamespace

import pytest


def test_editors_use_current_preferences_on_first_display(monkeypatch):
    from Qt.QtWidgets import QApplication
    from chimerax.cmd_line import settings as module
    from chimerax.ui.gui import MainWindow

    app = QApplication.instance() or QApplication([])
    if not isinstance(app, QApplication):
        pytest.skip('Requires a GUI application')
    settings = module._CmdLineSettings(None, 'deferred-command-settings-test')
    monkeypatch.setattr(module, 'settings', settings)
    window = SimpleNamespace(_settings_ui_widget=None, _accumulated_settings_options=[])
    window.add_settings_option = lambda *args, **kw: MainWindow.add_settings_option(window, *args, **kw)
    session = SimpleNamespace(ui=SimpleNamespace(main_window=window))
    module.register_settings_options(session)
    assert len(window._accumulated_settings_options) == 2
    assert all(factory for category, option, factory in window._accumulated_settings_options)
    settings.startup_commands = ['echo new startup command']
    settings.default_side = 'right'
    options = []
    panel = SimpleNamespace(add_option=lambda category, option: options.append(option))
    MainWindow._add_accumulated_settings_options(window, panel)
    try:
        by_name = {opt.attr_name: opt for opt in options}
        assert by_name['startup_commands'].value == ['echo new startup command']
        assert by_name['default_side'].value == 'right'
    finally:
        for option in options:
            option.widget.deleteLater()
