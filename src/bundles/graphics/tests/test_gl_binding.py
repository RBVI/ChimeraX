"""The ctypes binding in gl.py must stand in for PyOpenGL wherever ChimeraX uses it."""

import os
import re

import pytest


def _referenced_names():
    """GL.* names used by the graphics bundle's OpenGL module."""
    from chimerax.graphics import opengl
    with open(opengl.__file__, encoding="utf-8") as f:
        source = f.read()
    return set(re.findall(r"\bGL\.((?:gl|GL_)[A-Za-z0-9_]+)", source))


# Used only for debugging, so they are left to PyOpenGL.
_DELEGATED = {"glGetDebugMessageLog"}


def test_names_used_by_opengl_module_are_defined_without_pyopengl():
    from chimerax.graphics import gl
    missing = sorted(n for n in _referenced_names() - _DELEGATED if n not in vars(gl))
    assert missing == []


def test_constants_match_pyopengl():
    pyopengl = pytest.importorskip("OpenGL.GL")
    from chimerax.graphics import gl
    names = [n for n in vars(gl) if n.startswith("GL_")]
    assert names
    assert {n: getattr(gl, n) for n in names} == {n: int(getattr(pyopengl, n)) for n in names}


def test_functions_have_signatures():
    from chimerax.graphics import gl
    functions = [n for n in vars(gl) if n.startswith("gl") and callable(vars(gl)[n])]
    unknown = sorted(n for n in functions
                     if n not in gl._SIGNATURES and n not in ("glGetInteger", "glGetBoolean"))
    assert unknown == []


def test_other_names_come_from_pyopengl(monkeypatch):
    from types import SimpleNamespace
    from chimerax.graphics import gl
    monkeypatch.setattr(gl, "_pyopengl", lambda: SimpleNamespace(glFogf="pyopengl glFogf"))
    monkeypatch.delitem(vars(gl), "glFogf", raising=False)
    assert gl.glFogf == "pyopengl glFogf"


def test_unhandled_argument_forms_go_to_pyopengl(monkeypatch):
    from types import SimpleNamespace
    from chimerax.graphics import gl
    calls = []
    fake = SimpleNamespace(
        glGetIntegerv=lambda *args: calls.append(("glGetIntegerv", args)) or "array",
        glTexImage2D=lambda *args: calls.append(("glTexImage2D", args)),
    )
    monkeypatch.setattr(gl, "_pyopengl", lambda: fake)
    # A query with more than one value uses PyOpenGL's size table.
    assert gl.glGetIntegerv(gl.GL_POINT_SIZE_RANGE) == "array"
    # Lists are converted by PyOpenGL according to the pixel type.
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, 1, 1, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, [0, 0, 0, 0])
    assert [c[0] for c in calls] == ["glGetIntegerv", "glTexImage2D"]


def test_pointer_arguments():
    import ctypes
    import numpy
    from chimerax.graphics import gl
    assert gl._pointer(None) == (None, None)
    p = ctypes.c_void_p(16)
    assert gl._pointer(p) == (p, None)
    assert gl._pointer(32) == (32, None)
    a = numpy.arange(12, dtype=numpy.uint8).reshape(3, 4)[:, ::2]	# not contiguous
    address, keep = gl._pointer(a)
    assert keep.flags.c_contiguous and (keep == a).all()
    assert address == keep.ctypes.data
    assert gl._pointer([1, 2])[0] is gl._UNHANDLED


def test_error_raises_gl_error(monkeypatch):
    from chimerax.graphics import gl
    errors = [gl.GL_INVALID_OPERATION]
    monkeypatch.setitem(gl._cfuncs, "glUseProgram", lambda program: None)
    monkeypatch.setitem(gl._cfuncs, "glGetError", lambda: errors.pop() if errors else gl.GL_NO_ERROR)
    with pytest.raises(gl.GLError, match="invalid operation"):
        gl.glUseProgram(3)
    gl.glUseProgram(3)		# No error the second time.


def test_qt_contexts_use_the_binding_and_osmesa_uses_pyopengl(monkeypatch):
    from chimerax.graphics import gl, opengl
    monkeypatch.setattr(opengl, "_initialized_pyopengl", False)
    monkeypatch.setattr(opengl, "GL", None)
    opengl._initialize_pyopengl()
    assert opengl.GL is gl

    pytest.importorskip("OpenGL.GL")
    monkeypatch.setattr(opengl, "_initialized_pyopengl", False)
    monkeypatch.setattr(opengl, "_configure_pyopengl_to_use_osmesa", lambda: None)
    opengl._initialize_pyopengl(offscreen=True)
    assert opengl.GL.__name__ == "OpenGL.GL"
