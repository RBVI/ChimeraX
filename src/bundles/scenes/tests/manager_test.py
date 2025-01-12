import pytest
import chimerax.scenes.triggers as triggers
from unittest.mock import patch

@pytest.fixture
def scene_manager(test_production_session):
    return test_production_session.scenes

def test_scene_exists(test_production_session, scene_manager):
    """
    Test that scene_exists returns True if the scene exists and False if it does not.
    """
    assert not scene_manager.scene_exists('test_scene') # test_scene does not exist yet
    scene_manager.save_scene('test_scene') # create test scene
    assert scene_manager.scene_exists('test_scene') # test_scene exists now
    scene_manager.delete_scene('test_scene') # delete test scene
    assert not scene_manager.scene_exists('test_scene') # test_scene does not exist anymore


def test_delete_scene(scene_manager):
    """
    Test that delete_scene deletes a scene by name and activates the DELETED trigger.
    """
    def handler(trigger_name, scene_name):
        assert trigger_name == triggers.DELETED
        assert scene_name == 'test_scene'
        handler.called = True

    handler.called = False
    del_handler = triggers.add_handler(triggers.DELETED, handler)

    scene_manager.save_scene('test_scene') # create test scene
    assert scene_manager.scene_exists('test_scene') # test_scene exists now

    scene_manager.delete_scene('test_scene') # delete test scene
    assert handler.called # ensure handler was called
    assert not scene_manager.scene_exists('test_scene') # test_scene does not exist anymore

    triggers.remove_handler(del_handler)