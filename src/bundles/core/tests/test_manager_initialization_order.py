"""UI overlap preserves the manager-before-custom bundle initialization contract."""
from types import SimpleNamespace

from chimerax.core.toolshed import Toolshed


def test_ui_manager_initialization_joins_before_custom_initialization():
    calls = []
    toolshed = Toolshed.__new__(Toolshed)
    toolshed._installed_bundle_info = []
    toolshed._init_managers = lambda session: calls.append('managers') or []
    toolshed._init_custom = lambda session: calls.append('custom') or []
    def initialize(toolshed):
        calls.append('ui')
        return toolshed._init_managers(session)
    session = SimpleNamespace(ui=SimpleNamespace(_initialize_managers=initialize))
    toolshed.bootstrap_bundles(session, False)
    assert calls == ['ui', 'managers', 'custom']


def test_headless_and_safe_mode_keep_existing_initialization_paths():
    calls = []
    toolshed = Toolshed.__new__(Toolshed)
    toolshed._installed_bundle_info = []
    toolshed._init_managers = lambda session: calls.append('managers') or []
    toolshed._init_custom = lambda session: calls.append('custom') or []
    session = SimpleNamespace(ui=SimpleNamespace())
    toolshed.bootstrap_bundles(session, False)
    assert calls == ['managers', 'custom']
    calls.clear()
    toolshed.bootstrap_bundles(session, True)
    assert calls == []
