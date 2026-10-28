"""Measurable base classes load with the commands and retain ABC registrations."""
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def fresh_exports():
    path = Path(__file__).parents[1] / 'src' / '__init__.py'
    spec = importlib.util.spec_from_file_location('distance_exports_example', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_discovery_and_unknown_names_do_not_load_commands(monkeypatch):
    module = fresh_exports()
    calls = []
    original = object()
    implementation = SimpleNamespace(SimpleMeasurable=original)
    monkeypatch.setattr(importlib, 'import_module', lambda *args: calls.append(args) or implementation)
    assert 'SimpleMeasurable' not in vars(module)
    assert set(module.__all__) <= set(dir(module))
    with pytest.raises(AttributeError, match='no attribute'):
        module.NoSuchMeasurement
    assert calls == []
    assert module.SimpleMeasurable is original
    assert module.SimpleMeasurable is original
    assert calls == [('.cmd', module.__name__)]


def test_public_classes_keep_identity_and_atom_virtual_subclass():
    from chimerax import dist_monitor
    from chimerax.atomic import Atom
    from chimerax.dist_monitor import cmd

    assert dist_monitor.SimpleMeasurable is cmd.SimpleMeasurable
    assert dist_monitor.ComplexMeasurable is cmd.ComplexMeasurable
    assert issubclass(Atom, dist_monitor.SimpleMeasurable)
    namespace = {}
    exec('from chimerax.dist_monitor import *', namespace)
    assert namespace['SimpleMeasurable'] is cmd.SimpleMeasurable
    assert namespace['ComplexMeasurable'] is cmd.ComplexMeasurable
    assert namespace['cmd'] is cmd
