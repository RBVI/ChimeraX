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


