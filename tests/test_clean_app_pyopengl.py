"""Distribution cleanup preserves PyOpenGL's rendering backends and helpers."""

from pathlib import Path
import importlib.util

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "utils/clean_app.py"


@pytest.fixture(scope="module")
def clean_app():
    spec = importlib.util.spec_from_file_location("clean_app", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path, data=b"binding"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_remove_unused_pyopengl(clean_app, tmp_path):
    opengl = tmp_path / "OpenGL"
    removed = []
    for base in (opengl, opengl / "raw"):
        for api in ("AGL", "GLE", "GLES1", "GLES2", "GLES3", "GLUT", "Tk"):
            removed.append(_write(base / api / "__init__.py"))
        for vendor in ("AMD", "APPLE", "ATI", "GREMEDY", "MESA", "NV", "SGIX"):
            removed.append(_write(base / "GL" / vendor / "extension.py"))
    removed.append(_write(opengl / "GLES2/__pycache__/__init__.cpython-314.pyc"))
    for name in ("freeglut64.vc14.dll", "gle32.vc9.dll", "GLUT32.DLL",
                 "opengle64.vc14.dll"):
        removed.append(_write(opengl / "DLLS" / name))

    kept = []
    for base in (opengl, opengl / "raw"):
        for name in ("GL", "GLU", "WGL", "WGL/EXT", "GLX", "GLX/NV",
                     "EGL", "EGL/EXT", "EGL/NV", "osmesa", "GL/VERSION",
                     "GL/ARB", "GL/KHR", "GL/EXT"):
            kept.append(_write(base / name / "__init__.py"))
    for name in ("__init__.py", "GL/shaders.py", "GL/selection.py",
                 "GL/feedback.py", "platform/darwin.py", "platform/win32.py",
                 "platform/glx.py", "platform/egl.py", "platform/osmesa.py",
                 "arrays/numpymodule.py", "DLLS/freeglut_COPYING.txt",
                 "DLLS/gle_COPYING", "DLLS/other.dll", "DLLS/opengl32.dll"):
        kept.append(_write(opengl / name))
    kept.append(_write(tmp_path / "OpenGL_accelerate/wrapper.so"))
    kept.append(_write(tmp_path / "pyopengl-3.1.10.dist-info/LICENSE.txt"))

    expected = sum(p.stat().st_size for p in removed)
    assert clean_app.remove_unused_pyopengl(str(tmp_path)) == expected
    assert all(not p.exists() for p in removed)
    assert not (opengl / "GLES2").exists()
    assert all(p.exists() for p in kept)
    assert clean_app.remove_unused_pyopengl(str(tmp_path)) == 0


def test_no_pyopengl(clean_app, tmp_path):
    assert clean_app.remove_unused_pyopengl(str(tmp_path)) == 0


def test_unused_directory_symlink_does_not_remove_target(clean_app, tmp_path):
    target = _write(tmp_path / "external/extension.py")
    opengl = tmp_path / "OpenGL"
    opengl.mkdir()
    link = opengl / "GLES2"
    link.symlink_to(target.parent, target_is_directory=True)

    expected = link.lstat().st_size
    assert clean_app.remove_unused_pyopengl(str(tmp_path)) == expected
    assert not link.is_symlink()
    assert target.exists()


@pytest.mark.parametrize("parent", ["OpenGL", "OpenGL/raw", "OpenGL/GL",
                                  "OpenGL/raw/GL", "OpenGL/DLLS"])
def test_parent_symlink_does_not_prune_external_tree(clean_app, tmp_path, parent):
    external = tmp_path / "external"
    kept = [_write(external / name) for name in (
        "GLES2/__init__.py", "GL/NV/extension.py", "NV/extension.py",
        "freeglut64.vc14.dll")]
    link = tmp_path / "site-packages" / parent
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(external, target_is_directory=True)

    assert clean_app.remove_unused_pyopengl(str(tmp_path / "site-packages")) == 0
    assert link.is_symlink()
    assert all(p.exists() for p in kept)
