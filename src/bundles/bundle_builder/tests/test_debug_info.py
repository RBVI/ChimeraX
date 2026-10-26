import importlib

import pytest


@pytest.fixture(params=["bundle_builder", "bundle_builder_toml"])
def builder(request):
    return importlib.import_module("chimerax.bundle_builder." + request.param)


class FakeCompiler:
    def __init__(self, compiler_type="unix"):
        self.compiler_type = compiler_type
        self.calls = []

    def compile(self, sources, **kw):
        self.calls.append((sources, kw))
        return sources


@pytest.mark.parametrize("compiler_type", ["unix", "cygwin", "mingw32"])
def test_no_debug_info_without_debug(builder, compiler_type):
    assert builder._debug_info_args(FakeCompiler(compiler_type), False) == ["-g0"]


@pytest.mark.parametrize("compiler_type", ["unix", "msvc"])
def test_debug_info_args_leave_debug_builds_and_msvc_alone(builder, compiler_type):
    assert builder._debug_info_args(FakeCompiler(compiler_type), True) == []
    if compiler_type == "msvc":
        assert builder._debug_info_args(FakeCompiler(compiler_type), False) == []


def test_c_library_compiles_get_the_debug_info_args(builder, tmp_path):
    source = tmp_path / "file.cpp"
    source.write_text("x")
    compiler = FakeCompiler()
    builder._compile_in_parallel(
        compiler,
        [(str(source), [])],
        debug=False,
        extra_postargs=builder._debug_info_args(compiler, False),
    )
    assert compiler.calls[0][1]["extra_postargs"] == ["-g0"]


def _build_extensions(builder, monkeypatch, debug):
    # Run _BuildExt.build_extensions with a base class that compiles the way
    # build_ext.build_extension does
    def build_extensions(self):
        self.compiler.compile(
            ["a.cpp"], macros=[], debug=debug, extra_postargs=["-O2"], depends=[]
        )

    monkeypatch.setattr(builder._build_ext, "build_extensions", build_extensions)
    command = object.__new__(builder._BuildExt)
    command.compiler = FakeCompiler()
    command.build_extensions()
    return command.compiler.calls


def test_c_module_compiles_leave_out_debug_info(builder, monkeypatch):
    calls = _build_extensions(builder, monkeypatch, debug=False)
    assert calls == [
        (
            ["a.cpp"],
            {
                "macros": [],
                "debug": False,
                "extra_preargs": None,
                "extra_postargs": ["-O2", "-g0"],
                "depends": [],
            },
        )
    ]


def test_c_module_debug_compiles_keep_debug_info(builder, monkeypatch):
    calls = _build_extensions(builder, monkeypatch, debug=True)
    assert calls[0][1]["extra_postargs"] == ["-O2"]


def test_c_module_sources_compile_in_their_own_calls(builder, monkeypatch, tmp_path):
    sources = []
    for name, size in (("a.cpp", 1), ("b.cpp", 100), ("c.cpp", 10)):
        path = tmp_path / name
        path.write_text("x" * size)
        sources.append(str(path))
    preargs = ["-std=c++17"]
    results = []

    def build_extensions(self):
        results.append(self.compiler.compile(
            sources, macros=[], debug=False, extra_preargs=preargs,
            extra_postargs=["-O2"], depends=[],
        ))

    monkeypatch.setattr(builder._build_ext, "build_extensions", build_extensions)
    command = object.__new__(builder._BuildExt)
    command.compiler = FakeCompiler()
    command.build_extensions()
    calls = command.compiler.calls
    # One call per source, each with -g0 once, returning objects in order
    assert sorted(s for (s,), _ in calls) == sorted(sources)
    assert all(kw["extra_postargs"] == ["-O2", "-g0"] for _, kw in calls)
    assert all(kw["extra_preargs"] == preargs for _, kw in calls)
    # each has its own copy, which MSVC's compile() appends to
    assert len({id(kw["extra_preargs"]) for _, kw in calls}) == len(sources)
    assert all(kw["extra_preargs"] is not preargs for _, kw in calls)
    assert results == [sources]
