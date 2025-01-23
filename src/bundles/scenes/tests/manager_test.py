import pytest
import chimerax.scenes.triggers as scene_triggers
from unittest.mock import patch

@pytest.fixture
def scene_manager(test_production_session):
    return test_production_session.scenes

def setup_trigger_handler(trigger_type, expected_scene_name):
    """
    Helper function to set up a trigger handler for testing if the scene manager activates triggers correctly.

    Args:
        trigger_type (str): The name of the trigger to set up the handler for.
        expected_scene_name (str): The expected scene name to be passed to the handler.

    Returns:
        tuple: A tuple containing the handler function and the trigger handler. The handler function has the attribute
                `called` which is initialized to False and is set to True when the handler is called.
    """
    def handler_func(trigger_name, scene_name):
        assert trigger_name == trigger_type
        assert scene_name == expected_scene_name
        handler_func.called = True

    handler_func.called = False
    trigger_handler = scene_triggers.add_handler(trigger_type, handler_func)
    return handler_func, trigger_handler

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
    handler_func, trigger_handler = setup_trigger_handler(scene_triggers.DELETED, 'test_scene')

    scene_manager.save_scene('test_scene') # create test scene
    assert scene_manager.scene_exists('test_scene') # test_scene exists now

    scene_manager.delete_scene('test_scene') # delete test scene
    assert handler_func.called # ensure handler was called
    assert not scene_manager.scene_exists('test_scene') # test_scene does not exist anymore

    triggers.remove_handler(del_handler)
    scene_triggers.remove_handler(trigger_handler)