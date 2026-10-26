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

