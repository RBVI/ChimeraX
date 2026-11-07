"""Choose the faster bundled macOS style without overriding caller choices."""
from types import SimpleNamespace

import pytest


@pytest.fixture
def policy(monkeypatch):
    from chimerax.ui import gui, desktop_settings
    calls = []
    state = {'instance': None}

    class Application:
        @staticmethod
        def instance():
            return state['instance']

        @staticmethod
        def setStyle(name):
            assert state['instance'] is None
            calls.append(('style', name))

    monkeypatch.setattr(gui, 'QApplication', Application)
    monkeypatch.setattr(gui, 'initialize_qt_plugins_location', lambda: calls.append('plugins'))
    monkeypatch.setattr(gui, 'initialize_shared_opengl_contexts', lambda: calls.append('sharing'))
    monkeypatch.setattr(desktop_settings, 'select_platform_theme', lambda: calls.append('theme'))
    monkeypatch.setattr(gui.sys, 'platform', 'darwin')
    monkeypatch.delenv('QT_STYLE_OVERRIDE', raising=False)
    return SimpleNamespace(gui=gui, calls=calls, state=state)


def test_mac_default_selects_fusion_after_platform_configuration(policy):
    policy.gui.initialize_qt()
    assert policy.calls == ['plugins', 'sharing', 'theme', ('style', 'Fusion')]


def test_empty_override_uses_the_default(policy, monkeypatch):
    monkeypatch.setenv('QT_STYLE_OVERRIDE', '')
    policy.gui.initialize_qt()
    assert policy.calls[-1] == ('style', 'Fusion')


@pytest.mark.parametrize('style', ['macos', 'Fusion', 'custom-style'])
def test_explicit_style_choice_is_preserved(policy, monkeypatch, style):
    monkeypatch.setenv('QT_STYLE_OVERRIDE', style)
    policy.gui.initialize_qt()
    assert policy.calls == ['plugins', 'sharing', 'theme']


def test_existing_application_style_is_preserved(policy):
    policy.state['instance'] = object()
    policy.gui.initialize_qt()
    assert policy.calls == ['plugins', 'sharing', 'theme']


@pytest.mark.parametrize('platform', ['win32', 'linux'])
def test_other_platforms_keep_their_current_style_policy(policy, monkeypatch, platform):
    monkeypatch.setattr(policy.gui.sys, 'platform', platform)
    policy.gui.initialize_qt()
    assert policy.calls == ['plugins', 'sharing', 'theme']
