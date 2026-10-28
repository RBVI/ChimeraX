"""Third-party patches in prereqs/pips that keep modules out of ChimeraX
startup without changing behavior.  Each check runs in a fresh interpreter so
that it sees what importing the module alone does."""

import subprocess
import sys
import textwrap

import pytest


def _run(code, **env):
    import os
    result = subprocess.run([sys.executable, "-I", "-W", "always", "-c", textwrap.dedent(code)],
                            capture_output=True, text=True, env=dict(os.environ, **env))
    if result.returncode != 0:
        if "ModuleNotFoundError" in result.stderr:
            pytest.skip(result.stderr.strip().splitlines()[-1])
        raise AssertionError(result.stderr)
    return result.stdout.split()


def test_numpy_docstrings_deferred_on_request():
    out = _run("""
        import sys
        sys.numpy_defer_docstrings = True
        import numpy
        if not hasattr(numpy._core, "_load_docstrings"):
            print("unpatched")
            raise SystemExit
        print(numpy.zeros.__doc__ is None, "textwrap" in sys.modules)
        numpy._core._load_docstrings()
        print(numpy.zeros.__doc__.startswith("zeros("), numpy.float64.__doc__ is not None)
    """)
    if out == ["unpatched"]:
        pytest.skip("numpy_lazy_docstrings.patch not applied")
    assert out == ["True", "False", "True", "True"]


def test_numpy_docstrings_by_default():
    out = _run("""
        import numpy
        print(numpy.zeros.__doc__.startswith("zeros("))
    """)
    assert out == ["True"]


def test_numpy_does_not_import_platform():
    out = _run("""
        import sys, numpy
        print("platform" in sys.modules)
    """)
    assert out == ["False"]


def test_sortedcontainers_comparison_docs():
    out = _run("""
        import sys, sortedcontainers as sc
        print("textwrap" in sys.modules)
        for cls in (sc.SortedList, sc.SortedSet):
            doc = cls.__eq__.__doc__
            print(doc.startswith("Return true if and only if sorted"), doc.endswith("`other`\\n\\n"))
    """)
    assert out == ["False", "True", "True", "True", "True"]


def test_qt_keyboard_modifiers_without_textwrap():
    out = _run("""
        import sys, warnings
        from PySide6.QtCore import Qt, QKeyCombination
        print("textwrap" in sys.modules)
        shift, ctrl = Qt.KeyboardModifier.ShiftModifier, Qt.KeyboardModifier.ControlModifier
        print((shift | ctrl) == Qt.KeyboardModifier(shift.value | ctrl.value))
        print(isinstance(shift | Qt.Key.Key_A, QKeyCombination))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            print(shift + ctrl == shift | ctrl, "deprecated" in str(w[0].message))
    """, QT_QPA_PLATFORM="offscreen")
    if out[0] == "True":
        pytest.skip("patch_pyside6.py patch_qtcore_snippet not applied")
    assert out == ["False", "True", "True", "True", "True"]
