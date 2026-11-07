"""The Windows check for Intel drivers that blank Qt panels (ChimeraX bug
#2537) must not import HtmlView, and with it QtWebEngine, on other GPUs."""
import sys
from types import SimpleNamespace

import pytest


def _window(vendor, version):
    render = SimpleNamespace(opengl_vendor=lambda: vendor, opengl_version=lambda: version)
    warnings = []
    logger = SimpleNamespace(warning=warnings.append)
    return SimpleNamespace(view=SimpleNamespace(render=render),
                           session=SimpleNamespace(logger=logger)), warnings


@pytest.fixture
def check(monkeypatch):
    graphics = pytest.importorskip("chimerax.ui.graphics")
    monkeypatch.setattr(sys, "platform", "win32")
    return graphics.GraphicsWindow._check_for_bad_intel_driver


def test_other_gpus_do_not_import_htmlview(check, monkeypatch):
    monkeypatch.delitem(sys.modules, "chimerax.ui.widgets.htmlview", raising=False)
    window, warnings = _window("NVIDIA Corporation", "4.6.0 NVIDIA 580.0")
    check(window)
    assert "chimerax.ui.widgets.htmlview" not in sys.modules
    assert warnings == []


def test_affected_intel_driver_gets_native_windows(check, monkeypatch):
    from chimerax.ui import widgets
    fake = type("HtmlView", (), {"require_native_window": False})
    monkeypatch.setattr(widgets, "HtmlView", fake, raising=False)
    window, warnings = _window("Intel", "4.6.0 - Build 26.20.100.7000")
    check(window)
    assert fake.require_native_window is True
    assert len(warnings) == 1
    check(window)		# Applied once.
    assert len(warnings) == 1
