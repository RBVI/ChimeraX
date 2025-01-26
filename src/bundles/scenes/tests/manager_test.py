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
        if scene_name == expected_scene_name:
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

    scene_triggers.remove_handler(trigger_handler)

def test_edit_scene(scene_manager):
    """
    Test that edit_scene re-initializes the scene from the current session state and activates the EDITED trigger.
    """
    handler_func, trigger_handler = setup_trigger_handler(scene_triggers.EDITED, 'test_scene')

    scene_manager.save_scene('test_scene') # create test scene
    assert scene_manager.scene_exists('test_scene') # test_scene exists now

    scene_manager.edit_scene('test_scene') # edit test scene
    assert handler_func.called # ensure handler was called

    scene_triggers.remove_handler(trigger_handler)

def test_save_scene(scene_manager):
    """
    Test that save_scene saves the current state as a scene and activates the SAVED trigger.
    """
    handler_func, trigger_handler = setup_trigger_handler(scene_triggers.SAVED, 'test_scene')

    assert not scene_manager.scene_exists('test_scene') # test_scene does not exist yet
    scene_manager.save_scene('test_scene') # save test scene
    assert handler_func.called # ensure handler was called
    assert scene_manager.scene_exists('test_scene') # test_scene exists now

    scene_triggers.remove_handler(trigger_handler)

def test_clear_scenes(scene_manager):
    """
    Test that clear deletes all scenes from the scene manager and the DELETED trigger is activated for each scene.
    """
    handler_func1, trigger_handler1 = setup_trigger_handler(scene_triggers.DELETED, 'test_scene_1')
    handler_func2, trigger_handler2 = setup_trigger_handler(scene_triggers.DELETED, 'test_scene_2')

    scene_manager.save_scene('test_scene_1') # create first test scene
    scene_manager.save_scene('test_scene_2') # create second test scene
    assert scene_manager.scene_exists('test_scene_1') # test_scene_1 exists now
    assert scene_manager.scene_exists('test_scene_2') # test_scene_2 exists now

    scene_manager.clear() # clear all scenes
    assert not scene_manager.scene_exists('test_scene_1') # test_scene_1 does not exist anymore
    assert not scene_manager.scene_exists('test_scene_2') # test_scene_2 does not exist anymore
    assert handler_func1.called # ensure handler for test_scene_1 was called
    assert handler_func2.called # ensure handler for test_scene_2 was called

    scene_triggers.remove_handler(trigger_handler1)
    scene_triggers.remove_handler(trigger_handler2)