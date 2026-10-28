"""An empty scene manager defers scene capture code until save or restore."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def manager(monkeypatch):
    path = Path(__file__).parents[1] / 'src' / 'manager.py'
    spec = importlib.util.spec_from_file_location('chimerax.scenes._lazy_manager_example', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert 'Scene' not in vars(module)
    events = []
    monkeypatch.setattr(module, 'activate_trigger', lambda *args: events.append(args))

    class ExampleScene:
        def __init__(self, session, name):
            self.name = name
            self.updates = 0
        def get_name(self):
            return self.name
        def init_from_session(self):
            self.updates += 1
        @staticmethod
        def restore_snapshot(session, data):
            return ExampleScene(session, data['name'])

    monkeypatch.setitem(sys.modules, 'chimerax.scenes.scene', SimpleNamespace(Scene=ExampleScene))
    registrations = []
    session = SimpleNamespace(models=[], triggers=SimpleNamespace(
        add_handler=lambda *args: registrations.append(args)))
    instance = module.SceneManager(session)
    assert instance.get_scene_names() == []
    assert instance.get_scene('missing') is None
    assert instance.take_snapshot(session, None)['scenes'] == []
    assert 'Scene' not in vars(module)
    return instance, ExampleScene, events


def test_first_save_creates_scene_and_resaving_updates_it(manager):
    instance, scene_type, events = manager
    instance.save_scene('first')
    scene = instance.get_scene('first')
    assert isinstance(scene, scene_type)
    instance.save_scene('first')
    assert instance.get_scene('first') is scene
    assert scene.updates == 1
    assert instance.num_saved_scenes == 1
    assert len(events) == 2


def test_first_restore_creates_scenes_and_restores_saved_count(manager):
    instance, scene_type, events = manager
    instance._restore_snapshot({'scenes': [{'name': 'restored'}], 'num_saved_scenes': 7})
    assert isinstance(instance.get_scene('restored'), scene_type)
    assert instance.get_scene_names() == ['restored']
    assert instance.num_saved_scenes == 7
