from types import SimpleNamespace

import pytest


def test_legacy_editor_exports_are_discoverable_and_cached(monkeypatch):
    from chimerax.toolbar import tool, settings_ui
    for name in tool._settings_ui_exports:
        monkeypatch.setitem(tool.__dict__, name, object())
        monkeypatch.delitem(tool.__dict__, name)
        assert name in dir(tool)
        assert getattr(tool, name) is getattr(settings_ui, name)
        assert tool.__dict__[name] is getattr(settings_ui, name)
        if not name.startswith('_'):
            assert name in tool.__all__
    with pytest.raises(AttributeError):
        tool.no_such_editor_export


def test_editor_uses_current_toolbar_settings(monkeypatch):
    from chimerax.toolbar import tool, settings_ui
    calls = []
    # Importing the editor before toolbar settings are initialized must not
    # capture None or retain an obsolete settings object.
    monkeypatch.setattr(tool, '_settings', None)
    editor = SimpleNamespace(update_from_settings=lambda: calls.append('refresh'))
    for label in ('first', 'replacement'):
        current = SimpleNamespace(
            save=lambda label=label: calls.append((label, 'save')),
            reset=lambda label=label: calls.append((label, 'reset')),
            restore=lambda label=label: calls.append((label, 'restore')),
        )
        monkeypatch.setattr(tool, '_settings', current)
        settings_ui.ToolbarSettingsTool.save(editor)
        settings_ui.ToolbarSettingsTool.reset(editor)
        settings_ui.ToolbarSettingsTool.restore(editor)
    assert calls == [
        ('first', 'save'), ('first', 'reset'), 'refresh',
        ('first', 'restore'), 'refresh',
        ('replacement', 'save'), ('replacement', 'reset'), 'refresh',
        ('replacement', 'restore'), 'refresh',
    ]
