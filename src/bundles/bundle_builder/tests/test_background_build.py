import importlib
import os
import threading

import pytest


@pytest.fixture(params=["bundle_builder", "bundle_builder_toml"])
def builder(request):
    return importlib.import_module("chimerax.bundle_builder." + request.param)


class FakeBackground:
    def __init__(self, events):
        self.events = events

    def wait(self):
        self.events.append("wait")


class FakeCompiler:
    compiler_type = "unix"

    def __init__(self, events):
        self.events = events

    def compile(self, sources, **kw):
        self.events.append("compile")
        return sources

    def link_shared_object(self, *args, **kw):
        self.events.append("link")


def _command(builder, background=None):
    cls = type("build_ext", (builder._BuildExt,), {"background": background})
    return object.__new__(cls)


def test_build_runs_build_ext_first(builder):
    names = [name for name, _ in builder._Build.sub_commands]
    assert names[0] == "build_ext"
    assert sorted(names) == sorted(name for name, _ in builder._build.sub_commands)
    assert names.index("build_py") > 0


def test_background_build_runs_alongside_the_caller(builder):
    started = threading.Event()
    release = threading.Event()

    def work():
        started.set()
        release.wait(10)

    background = builder._BackgroundBuild(work)
    background.start()
    assert started.wait(10)
    release.set()
    background.wait()


def test_background_build_error_reaches_wait(builder):
    def work():
        raise ValueError("compile failed")

    background = builder._BackgroundBuild(work)
    background.start()
    with pytest.raises(RuntimeError) as info:
        background.wait()
    assert isinstance(info.value.__cause__, ValueError)


def test_background_build_join_before_start_does_nothing(builder):
    builder._BackgroundBuild(lambda: None).join()


def test_c_module_links_wait_for_the_libraries(builder, monkeypatch):
    events = []

    def build_extensions(self):
        self.compiler.compile(["a.cpp"], debug=False)
        self.compiler.link_shared_object(["a.o"], "a.so")

    monkeypatch.setattr(builder._build_ext, "build_extensions", build_extensions)
    command = _command(builder, FakeBackground(events))
    command.compiler = FakeCompiler(events)
    command.build_extensions()
    assert events == ["compile", "wait", "link"]


def test_c_module_links_do_not_wait_without_libraries(builder, monkeypatch):
    events = []

    def build_extensions(self):
        self.compiler.link_shared_object(["a.o"], "a.so")

    monkeypatch.setattr(builder._build_ext, "build_extensions", build_extensions)
    command = _command(builder)
    command.compiler = FakeCompiler(events)
    command.build_extensions()
    assert events == ["link"]


def test_build_ext_waits_for_the_libraries_before_build_py(builder, monkeypatch):
    events = []
    monkeypatch.setattr(builder._build_ext, "run", lambda self: events.append("run"))
    _command(builder, FakeBackground(events)).run()
    assert events == ["run", "wait"]


def test_each_extension_has_its_own_object_folder(builder, monkeypatch):
    class Ext:
        name = "chimerax.example._ext"

    seen = []
    monkeypatch.setattr(
        builder._build_ext,
        "build_extension",
        lambda self, ext: seen.append(self.build_temp),
    )
    command = _command(builder)
    command.build_temp = os.path.join("build", "temp")
    command.build_extension(Ext())
    assert seen == [os.path.join("build", "temp", Ext.name)]
    # The command the build_ext -j threads share is unchanged
    assert command.build_temp == os.path.join("build", "temp")


class FakeLibrary:
    static = False

    def __init__(self, name, events, barrier):
        self.name = name
        self.events = events
        self.barrier = barrier

    def compile_objects(self, logger, dependencies, static, debug):
        # Every library's compile waits for the others, so this only
        # finishes if they compile at once
        self.barrier.wait()
        return "compiler", [self.name + ".o"], []

    def link(self, compiler, objs, extra_link_args, debug=False):
        self.events.append(("link", self.name, objs))


def test_libraries_compile_at_once_then_link_in_order(builder):
    events = []
    barrier = threading.Barrier(3, timeout=10)
    libraries = [FakeLibrary(name, events, barrier) for name in ("a", "b", "c")]
    builder._compile_libraries(libraries, None, [])
    assert events == [("link", "a", ["a.o"]), ("link", "b", ["b.o"]),
                      ("link", "c", ["c.o"])]


def test_one_library_compiles_as_before(builder):
    calls = []

    class Library:
        def compile(self, logger, dependencies, debug=False):
            calls.append(debug)

    builder._compile_libraries([Library()], None, [], debug=True)
    assert calls == [True]
