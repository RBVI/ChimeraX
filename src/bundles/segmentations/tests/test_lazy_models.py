"""Startup registers segmentation hooks without loading the volume model."""
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_source(name, relative_path):
    path = Path(__file__).parents[1] / 'src' / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_package_discovery_and_resolution_preserve_public_identity(monkeypatch):
    module = load_source('segmentation_exports_example', '__init__.py')
    assert 'Segmentation' not in vars(module)
    assert 'open_grids_as_segmentation' not in vars(module)
    assert set(module.__all__) <= set(dir(module))
    calls = []
    original = object()
    implementation = SimpleNamespace(Segmentation=original)
    monkeypatch.setattr(importlib, 'import_module', lambda *args: calls.append(args) or implementation)
    with pytest.raises(AttributeError, match='no attribute'):
        module.NoSuchSegmentation
    assert calls == []
    assert module.Segmentation is original
    assert module.Segmentation is original
    assert calls == [('.segmentation', module.__name__)]


def test_tracker_registration_keeps_model_types_unresolved(monkeypatch):
    module = load_source('segmentation_tracker_example', 'segmentation_tracker.py')
    assert 'Volume' not in vars(module)
    assert 'Segmentation' not in vars(module)
    registrations = []
    session = SimpleNamespace(triggers=SimpleNamespace(
        add_handler=lambda *args: registrations.append(args)))
    module.register_model_trigger_handlers(session)
    from chimerax.core.models import ADD_MODELS, REMOVE_MODELS
    assert [name for name, callback in registrations] == [ADD_MODELS, REMOVE_MODELS]
    assert module.get_tracker() is module.get_tracker()
    assert module.get_tracker().active_segmentation is None
    assert 'Volume' not in vars(module)
    assert 'Segmentation' not in vars(module)


def test_package_exports_and_session_restoration_return_the_original_class():
    import chimerax.segmentations as package
    from chimerax.segmentations.segmentation import Segmentation, open_grids_as_segmentation
    assert package.Segmentation is Segmentation
    assert package.open_grids_as_segmentation is open_grids_as_segmentation
    assert package.bundle_api.get_class('Segmentation') is Segmentation
    assert package.bundle_api.get_class('NoSuchClass') is None


def test_first_model_callbacks_track_volume_and_segmentation_lifetimes(monkeypatch):
    import chimerax.map as maps
    import chimerax.segmentations.segmentation as models
    import chimerax.segmentations.triggers as triggers

    module = load_source('segmentation_tracker_callbacks_example', 'segmentation_tracker.py')
    class ExampleVolume:
        pass
    class ExampleSegmentation:
        def __init__(self, reference_volume):
            self.reference_volume = reference_volume
            self.active = False
    monkeypatch.setattr(maps, 'Volume', ExampleVolume)
    monkeypatch.setattr(models, 'Segmentation', ExampleSegmentation)
    events = []
    monkeypatch.setattr(triggers, 'activate_trigger', lambda *args: events.append(args))
    volume = ExampleVolume()
    segmentation = ExampleSegmentation(volume)
    session = SimpleNamespace()
    module.on_model_added_to_session(session, None, [volume, segmentation])
    tracker = module.get_tracker()
    assert tracker.segmentations_for_volume(volume) == {segmentation}
    tracker.active_segmentation = segmentation
    assert segmentation.active
    module.on_model_removed_from_session(session, None, [segmentation])
    assert tracker.active_segmentation is None
    assert not segmentation.active
    assert tracker.segmentations_for_volume(volume) == set()
    module.on_model_removed_from_session(session, None, [volume])
    assert volume not in tracker
