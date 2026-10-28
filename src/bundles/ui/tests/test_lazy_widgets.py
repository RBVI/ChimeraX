import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_exports():
    path = Path(__file__).parents[1] / 'src' / 'widgets' / '__init__.py'
    spec = importlib.util.spec_from_file_location('widget_exports_example', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_one_widget_does_not_load_other_implementations(monkeypatch):
    widgets = load_exports()
    calls = []
    widget = object()

    def import_module(name, package):
        calls.append((name, package))
        return SimpleNamespace(Citation=widget)

    monkeypatch.setattr('importlib.import_module', import_module)
    assert widgets.Citation is widget
    assert widgets.Citation is widget
    assert calls == [('.citation', widgets.__name__)]
    assert 'ItemTable' not in vars(widgets)
    assert 'ItemTable' in dir(widgets)
    assert 'ItemTable' in widgets.__all__
    with pytest.raises(AttributeError):
        widgets.missing_widget


def test_module_exports_and_html_names_remain_available(monkeypatch):
    widgets = load_exports()
    implementation = SimpleNamespace(HtmlView=object(), ChimeraXHtmlView=object())
    monkeypatch.setattr('importlib.import_module', lambda *args: implementation)
    assert widgets.citation is implementation
    assert widgets.HtmlView is implementation.HtmlView
    assert widgets.ChimeraXHtmlView is implementation.ChimeraXHtmlView
    assert 'citation' in widgets.__all__
