"""Command help searches defaults before locating bundle documentation."""
from types import SimpleNamespace

import pytest

from chimerax.core import toolshed
from chimerax.core.commands import cli


@pytest.fixture
def help_paths(monkeypatch, tmp_path):
    defaults = [tmp_path / 'cache', tmp_path / 'builtin']
    bundles = [tmp_path / 'bundle1', tmp_path / 'bundle2']
    calls = []

    class Installed:
        @property
        def help_directories(self):
            calls.append('bundle paths')
            return [str(p) for p in bundles]

    monkeypatch.setattr(toolshed, '_default_help_dirs', [str(p) for p in defaults])
    monkeypatch.setattr(toolshed, '_toolshed', SimpleNamespace(_installed_bundle_info=Installed()))
    return defaults, bundles, calls


def page(directory, name):
    path = directory / 'user' / 'commands' / (name + '.html')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('command help')


@pytest.mark.parametrize('index', [0, 1])
def test_default_command_help_skips_bundle_scan(help_paths, index):
    defaults, bundles, calls = help_paths
    page(defaults[index], 'example')
    assert cli._get_help_url(['example', 'subcommand']) == 'help:user/commands/example.html#subcommand'
    assert cli._get_help_url(['~example']) == 'help:user/commands/example.html#~example'
    assert calls == []


def test_bundle_help_fallback_and_missing_page(help_paths):
    defaults, bundles, calls = help_paths
    page(bundles[1], 'plugin')
    assert cli._get_help_url(['plugin']) == 'help:user/commands/plugin.html'
    assert calls == ['bundle paths']
    assert cli._get_help_url(['absent']) is None
    assert calls == ['bundle paths', 'bundle paths']


def test_public_help_directories_preserves_order_and_returns_fresh_list(help_paths):
    defaults, bundles, calls = help_paths
    expected = [str(p) for p in defaults + bundles]
    paths = toolshed.get_help_directories()
    assert paths == expected
    paths.clear()
    assert toolshed.get_help_directories() == expected
    assert calls == ['bundle paths', 'bundle paths']


def test_help_paths_before_toolshed_initialization(help_paths, monkeypatch):
    defaults, bundles, calls = help_paths
    monkeypatch.setattr(toolshed, '_toolshed', None)
    page(defaults[1], 'core')
    assert cli._get_help_url(['core']) == 'help:user/commands/core.html'
    assert cli._get_help_url(['absent']) is None
    assert toolshed.get_help_directories() == [str(p) for p in defaults]
    assert calls == []
