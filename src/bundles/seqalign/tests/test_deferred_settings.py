"""Sequence preference controls are materialized from current settings."""
from types import SimpleNamespace

import pytest


def test_viewer_and_threshold_editors_wait_until_settings_opens(monkeypatch):
    from Qt.QtWidgets import QApplication
    from chimerax.seqalign import settings as module
    from chimerax.ui.gui import MainWindow

    app = QApplication.instance() or QApplication([])
    if not isinstance(app, QApplication):
        pytest.skip('Requires a GUI application')
    settings = module._AlignmentsSettings(None, 'deferred-sequence-settings-test')
    monkeypatch.setattr(module, 'settings', settings)
    viewers = {'Sequence Viewer': ['seqview'], 'Alignment Grid': ['grid']}
    alignments = SimpleNamespace(registered_viewers=lambda variety: list(viewers),
        viewer_info={'alignment': viewers, 'sequence': viewers})
    window = SimpleNamespace(_settings_ui_widget=None, _accumulated_settings_options=[])
    window.add_settings_option = lambda *args, **kw: MainWindow.add_settings_option(window, *args, **kw)
    session = SimpleNamespace(ui=SimpleNamespace(main_window=window), alignments=alignments)
    module.register_settings_options(session)
    assert len(window._accumulated_settings_options) == 5
    assert all(factory for category, option, factory in window._accumulated_settings_options)
    settings.large_align_threshold = 500
    settings.small_align_viewer = 'grid'
    options = []
    panel = SimpleNamespace(add_option=lambda category, option: options.append(option))
    MainWindow._add_accumulated_settings_options(window, panel)
    try:
        by_name = {opt.attr_name: opt for opt in options}
        assert by_name['large_align_threshold'].value == 500
        assert by_name['small_align_viewer'].value == 'grid'
    finally:
        for option in options:
            option.widget.deleteLater()
