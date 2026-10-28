from types import SimpleNamespace

import pytest


def test_help_paths_are_cached_until_invalidated(monkeypatch, tmp_path):
    import chimerax
    from chimerax import help_viewer
    from urllib.request import pathname2url

    cache = tmp_path / 'docs' / 'user'
    cache.mkdir(parents=True)
    index = cache / 'index.html'
    index.write_text('cached index')
    monkeypatch.setattr(chimerax, 'app_dirs', SimpleNamespace(user_cache_dir=str(tmp_path)))
    monkeypatch.setattr(help_viewer, 'help_url_paths', [], raising=False)
    calls = []
    directory = str(tmp_path / 'α help')
    monkeypatch.setattr(help_viewer.toolshed, 'get_help_directories',
                        lambda: calls.append(True) or [directory])
    help_viewer._update_cache()
    assert not index.exists()
    assert not calls
    assert 'help_url_paths' in dir(help_viewer)
    assert 'help_url_paths' in help_viewer.__all__
    paths = help_viewer.help_url_paths
    expected = pathname2url(directory)
    if expected.startswith('///'):
        expected = expected[2:]
    assert paths == [expected.rstrip('/') + '/']
    assert help_viewer.help_url_paths is paths
    assert len(calls) == 1
    help_viewer._update_cache()
    assert len(calls) == 1
    assert help_viewer.help_url_paths == paths
    assert len(calls) == 2


def test_unknown_export_does_not_resolve_help_paths(monkeypatch):
    from chimerax import help_viewer
    def unexpected():
        raise AssertionError('Unknown attributes must not discover help directories')
    monkeypatch.setattr(help_viewer.toolshed, 'get_help_directories', unexpected)
    with pytest.raises(AttributeError, match='not_a_help_attribute'):
        help_viewer.__getattr__('not_a_help_attribute')
