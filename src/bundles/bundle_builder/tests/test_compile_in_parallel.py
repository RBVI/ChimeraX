import importlib
import threading
import time

import pytest


@pytest.fixture(params=["bundle_builder", "bundle_builder_toml"])
def builder(request):
    return importlib.import_module("chimerax.bundle_builder." + request.param)


class FakeCompiler:
    """Records the compile() calls _compile_in_parallel makes."""

    def __init__(self, on_compile=None):
        self.calls = []
        self.lock = threading.Lock()
        self.on_compile = on_compile

    def compile(self, sources, **kw):
        with self.lock:
            self.calls.append((sources, kw))
        if self.on_compile:
            self.on_compile(sources)
        return sources


def _sources(tmp_path, sizes):
    paths = []
    for i, size in enumerate(sizes):
        path = tmp_path / f"file{i}.cpp"
        path.write_text("x" * size)
        paths.append(str(path))
    return paths


def test_compiles_each_source_in_its_own_call(builder, tmp_path):
    cpp, c = _sources(tmp_path, [10, 20])
    compiler = FakeCompiler()
    builder._compile_in_parallel(
        compiler, [(cpp, ["-std=c++17"]), (c, [])], macros=[("A", 1)], debug=True
    )
    assert sorted(compiler.calls, key=lambda call: call[0]) == [
        ([cpp], {"extra_preargs": ["-std=c++17"], "macros": [("A", 1)], "debug": True}),
        ([c], {"extra_preargs": [], "macros": [("A", 1)], "debug": True}),
    ]


def test_largest_sources_start_first(builder, tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "cpu_count", 1)
    small, large, medium = _sources(tmp_path, [1, 100, 10])
    compiler = FakeCompiler()
    builder._compile_in_parallel(
        compiler, [(small, []), (large, []), (medium, []), ("missing.cpp", [])]
    )
    assert [call[0] for call in compiler.calls] == [
        [large],
        [medium],
        [small],
        ["missing.cpp"],
    ]


def test_sources_compile_at_the_same_time(builder, tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "cpu_count", 2)
    # Each compile waits for the other, so this only finishes if both run at once
    barrier = threading.Barrier(2, timeout=10)
    compiler = FakeCompiler(on_compile=lambda sources: barrier.wait())
    builder._compile_in_parallel(
        compiler, [(source, []) for source in _sources(tmp_path, [1, 2])]
    )
    assert len(compiler.calls) == 2


def test_failure_stops_the_remaining_compiles(builder, tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "cpu_count", 1)
    first, *rest = _sources(tmp_path, [100] + list(range(9, 0, -1)))

    def fail_first(sources):
        if sources == [first]:
            raise RuntimeError("compile failed")
        time.sleep(0.05)

    compiler = FakeCompiler(on_compile=fail_first)
    with pytest.raises(RuntimeError, match="compile failed"):
        builder._compile_in_parallel(
            compiler, [(source, []) for source in [first, *rest]]
        )
    # The worker may start the next source before the rest are cancelled
    assert compiler.calls[0][0] == [first]
    assert len(compiler.calls) <= 2


def test_uninitialized_compiler_is_initialized_once_first(builder, tmp_path):
    class LazyCompiler(FakeCompiler):
        # Like MSVCCompiler, which initializes itself on its first compile
        initialized = False
        initializations = 0

        def initialize(self):
            self.initializations += 1
            self.initialized = True

        def compile(self, sources, **kw):
            assert self.initialized
            return super().compile(sources, **kw)

    compiler = LazyCompiler()
    builder._compile_in_parallel(
        compiler, [(source, []) for source in _sources(tmp_path, [1, 2, 3])]
    )
    assert compiler.initializations == 1
    assert len(compiler.calls) == 3


def test_objects_are_returned_in_the_order_of_the_sources(builder, tmp_path):
    # Compiled largest first, but linked in the order given
    sources = _sources(tmp_path, [1, 100, 10])
    objs = builder._compile_in_parallel(FakeCompiler(), [(s, []) for s in sources])
    assert objs == sources
