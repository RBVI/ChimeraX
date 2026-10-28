from email.message import Message
from types import SimpleNamespace


def test_discovery_reads_classifiers_without_converting_metadata(monkeypatch):
    import importlib.metadata
    from chimerax.core.toolshed import installed

    class Metadata(Message):
        @property
        def json(self):
            raise AssertionError("Bundle discovery must not convert all metadata")

    class Distribution:
        def __init__(self, metadata):
            self.metadata = metadata

    def distribution(*classifiers):
        metadata = Metadata()
        metadata["Name"] = "example"
        metadata["Description"] = "Unrelated package description"
        for classifier in classifiers:
            metadata["Classifier"] = classifier
        return Distribution(metadata)

    bundle = distribution("Topic :: Scientific", "ChimeraX :: Bundle :: example")
    # Repeated discovery of the same bundle (e.g. editable installs) must
    # still be deduplicated, and distributions without classifiers skipped.
    duplicate = distribution("ChimeraX :: Bundle :: example")
    dists = [distribution(), distribution("Topic :: Scientific"), bundle, duplicate]
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: iter(dists))
    discovered = []
    info = SimpleNamespace(name="ChimeraX-Example")

    def make_info(dist, is_installed, logger, *, metadata):
        discovered.append(dist)
        assert is_installed
        assert metadata is dist.metadata
        return info

    monkeypatch.setattr(installed, "_make_bundle_info", make_info)
    monkeypatch.setattr(
        installed.InstalledBundleCache, "_order_bundles",
        lambda self, mapping, logger: list(mapping.values()),
    )
    cache = installed.InstalledBundleCache()
    cache.load(logger=None, write_cache=False)
    assert discovered == dists[2:]
    assert cache == [info]


def test_dependency_order_reuses_names_and_reads_requirements_once():
    from chimerax.core.toolshed.installed import InstalledBundleCache

    class Distribution:
        def __init__(self, requires):
            self._requires = requires
            self.reads = 0

        @property
        def name(self):
            raise AssertionError("Use the name already read into BundleInfo")

        @property
        def requires(self):
            self.reads += 1
            return self._requires

    a = Distribution(None)
    b = Distribution(["ChimeraX-A>=1", "numpy", "invalid requirement$"])
    c = Distribution(["ChimeraX-B"])
    info_a = SimpleNamespace(name="ChimeraX-A", inits={})
    info_b = SimpleNamespace(name="ChimeraX-B", inits={})
    info_c = SimpleNamespace(name="ChimeraX-C", inits={})
    bundles = {c: info_c, b: info_b, a: info_a}
    cache = InstalledBundleCache()
    assert cache._order_bundles(bundles, None) == [info_a, info_b, info_c]
    assert [a.reads, b.reads, c.reads] == [1, 1, 1]

    # An explicit initialization dependency suppresses the reverse package
    # dependency and must still break a dependency cycle as before.
    a._requires = ["ChimeraX-B"]
    info_a.inits = {"custom": ["ChimeraX-B"]}
    ordered = cache._order_bundles(bundles, None)
    assert ordered.index(info_b) < ordered.index(info_a)
    assert ordered.index(info_b) < ordered.index(info_c)


def test_cache_read_does_not_import_filelock(monkeypatch, tmp_path):
    import builtins
    import json
    import sys
    from chimerax.core.toolshed.installed import InstalledBundleCache

    executable = tmp_path / 'ChimeraX'
    executable.touch()
    monkeypatch.setattr(sys, 'executable', str(executable))
    cache_file = tmp_path / 'cache.json'
    cache_file.write_text(json.dumps([str(executable), executable.stat().st_mtime]) + '\n[]')
    monkeypatch.setattr(InstalledBundleCache, '_is_cache_newer', lambda *args: True)
    original_import = builtins.__import__

    def import_without_filelock(name, *args, **kw):
        if name == 'filelock':
            raise AssertionError('Cache hits must not import locking machinery')
        return original_import(name, *args, **kw)

    monkeypatch.setattr(builtins, '__import__', import_without_filelock)
    assert InstalledBundleCache()._read_cache(str(cache_file))


def test_atomic_cache_publish_keeps_old_snapshot_until_complete(monkeypatch, tmp_path):
    import json
    import sys
    from chimerax.core.toolshed.info import BundleInfo
    from chimerax.core.toolshed.installed import InstalledBundleCache

    executable = tmp_path / 'ChimeraX'
    executable.touch()
    monkeypatch.setattr(sys, 'executable', str(executable))
    monkeypatch.setattr(InstalledBundleCache, '_is_cache_newer', lambda *args: True)
    cache_file = tmp_path / 'cache.json'
    old = json.dumps([str(executable), executable.stat().st_mtime]) + '\n[]'
    cache_file.write_text(old)
    cache = InstalledBundleCache()
    cache.append(BundleInfo('ChimeraX-Example', installed=True, version='1'))
    dump = json.dump

    def inspect_during_write(*args, **kw):
        assert cache_file.read_text() == old
        assert InstalledBundleCache()._read_cache(str(cache_file))
        return dump(*args, **kw)

    monkeypatch.setattr(json, 'dump', inspect_during_write)
    errors = []
    cache._write_cache(str(cache_file), SimpleNamespace(error=errors.append))
    assert not errors
    restored = InstalledBundleCache()
    assert restored._read_cache(str(cache_file))
    assert [b.name for b in restored] == ['ChimeraX-Example']


def test_failed_serialization_preserves_previous_cache(monkeypatch, tmp_path):
    import pytest
    from chimerax.core.toolshed.installed import InstalledBundleCache

    cache_file = tmp_path / 'cache.json'
    cache_file.write_text('previous complete snapshot')
    cache = InstalledBundleCache()
    cache.append(SimpleNamespace(cache_data=lambda: object()))
    with pytest.raises(TypeError):
        cache._write_cache(str(cache_file), SimpleNamespace(error=lambda message: None))
    assert cache_file.read_text() == 'previous complete snapshot'
    assert not list(tmp_path.glob('*.tmp'))


def test_reader_validates_open_snapshot_timestamp(monkeypatch, tmp_path):
    import json
    import os
    import sys
    import pytest
    from chimerax.core.toolshed.installed import InstalledBundleCache

    if sys.platform.startswith('win'):
        pytest.skip('Windows does not allow replacing this open file; SaveTextFile retries instead')
    executable = tmp_path / 'ChimeraX'
    executable.touch()
    monkeypatch.setattr(sys, 'executable', str(executable))
    cache_file = tmp_path / 'cache.json'
    data = json.dumps([str(executable), executable.stat().st_mtime]) + '\n[]'
    cache_file.write_text(data)
    os.utime(cache_file, (100, 100))
    timestamps = []

    def newer(self, path, snapshot_time):
        replacement = tmp_path / 'replacement.json'
        replacement.write_text(data)
        os.utime(replacement, (200, 200))
        os.replace(replacement, path)
        timestamps.append(snapshot_time)
        return snapshot_time >= 150

    monkeypatch.setattr(InstalledBundleCache, '_is_cache_newer', newer)
    assert not InstalledBundleCache()._read_cache(str(cache_file))
    assert timestamps == [100]
    assert cache_file.stat().st_mtime == 200
