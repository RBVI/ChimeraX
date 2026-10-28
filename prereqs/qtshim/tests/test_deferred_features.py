"""PySide source classification waits until optional features are in use."""
from types import ModuleType
import sys

from Qt._features import defer_source_scans


def fake_feature(monkeypatch):
    feature = ModuleType('shibokensupport.feature')
    feature._is_initialized = False
    calls = []
    def original(module):
        calls.append(module)
    original.__module__ = feature.__name__
    feature.feature_imported = original
    monkeypatch.setitem(sys.modules, feature.__name__, feature)
    return feature, original, calls


def test_defers_then_restores_original_callback(monkeypatch):
    feature, original, calls = fake_feature(monkeypatch)
    defer_source_scans()
    callback = feature.feature_imported
    callback('before-feature-import')
    assert not calls
    # PySide initializes its full loaded-module registry on first feature use.
    feature._is_initialized = True
    callback('after-feature-import')
    assert calls == ['after-feature-import']
    assert feature.feature_imported is original


def test_leaves_existing_features_and_foreign_callbacks_alone(monkeypatch):
    feature, original, calls = fake_feature(monkeypatch)
    feature._is_initialized = True
    defer_source_scans()
    assert feature.feature_imported is original
    feature._is_initialized = False
    original.__module__ = 'another_integration'
    defer_source_scans()
    assert feature.feature_imported is original


def test_idempotent_and_unknown_internals_untouched(monkeypatch):
    feature, original, calls = fake_feature(monkeypatch)
    defer_source_scans()
    callback = feature.feature_imported
    defer_source_scans()
    assert feature.feature_imported is callback
    monkeypatch.delitem(sys.modules, feature.__name__)
    defer_source_scans()
    monkeypatch.setitem(sys.modules, feature.__name__, ModuleType(feature.__name__))
    defer_source_scans()


def test_real_feature_switching_in_fresh_process(tmp_path):
    """Check callbacks across shim, default-API, and opted-in modules."""
    import subprocess
    (tmp_path/'plain.py').write_text('''from PySide6.QtCore import QObject

def name(obj):
    obj.setObjectName("plain")
    return obj.objectName()
''')
    (tmp_path/'switched.py').write_text('''from PySide6.QtCore import QObject
from __feature__ import snake_case, true_property

def name(obj):
    obj.object_name = "switched"
    return obj.object_name
''')
    script = '''from Qt.QtCore import QObject
import plain
import shibokensupport.feature as feature
assert not feature._is_initialized
obj = QObject()
assert plain.name(obj) == "plain"
import switched
assert switched.name(obj) == "switched"
assert plain.name(obj) == "plain"
assert obj.objectName() == "plain"
assert feature._is_initialized
assert feature.feature_imported.__module__ == feature.__name__
assert feature.pyside_feature_dict['plain'] == 0
assert feature.pyside_feature_dict['switched'] == 3
'''
    subprocess.run([sys.executable, '-c', script], cwd=tmp_path, check=True)
