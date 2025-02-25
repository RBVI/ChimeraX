from chimerax.scenes import scene


class MockModelA:
    def restore_scene(self, data):
        pass

    def take_snapshot(self):
        pass

class MockModelB(MockModelA):
    pass

class MockModelC(MockModelB):
    def restore_scene(self, data):
        pass

    def take_snapshot(self):
        pass

class MockModelD:
    pass

class MockModelE():
    def take_snapshot(self):
        pass

class MockModelF():
    def restore_scene(self, data):
        pass

def test_md_scene_implementation():
    """
    Test that md_scene_implementation returns the most derived class that implements restore_scene.
    """
    model_a = MockModelA()
    model_b = MockModelB()
    model_c = MockModelC()
    model_d = MockModelD()
    model_e = MockModelE()
    model_f = MockModelF()

    # MockModelA implements restore_scene
    assert scene.md_scene_implementation(model_a) == MockModelA
    # MockModelB does not implement restore_scene, but it's super class does
    assert scene.md_scene_implementation(model_b) == MockModelA
    # MockModelC implements restore_scene
    assert scene.md_scene_implementation(model_c) == MockModelC
    # MockModelD does not implement restore_scene and neither do any of its super classes
    assert scene.md_scene_implementation(model_d) is None
    # Mock Model E does not implement restore_scene
    assert scene.md_scene_implementation(model_e) is None
    # Mock Model F does not implement take_snapshot
    assert scene.md_scene_implementation(model_f) is None

def test_scene_super():
    model_a = MockModelA()
    model_b = MockModelB()
    model_c = MockModelC()

    # Top level class does not have a super class that implements restore_scene
    assert scene.scene_super(model_a) is None
    # MockModelB does not implement scenes, but it's super class does
    assert scene.scene_super(model_b) == MockModelA
    # While MockModelC implements restore_scene, the closest parent class that implements scenes is MockModelA,
    # not MockModelB
    assert scene.scene_super(model_c) == MockModelA

""" Tests for the Scene class """

def test_init_from_session(test_production_session):
    scenes_mgr = test_production_session.scenes
    scenes_mgr.save_scene("test_scene")

    test_scene = scenes_mgr.get_scene("test_scene")
    test_scene.init_from_session()

    # Make sure version, name, thumbnail, main_view_data, named_view, and scene_models attributes are initialized
    assert hasattr(test_scene, 'version')
    assert hasattr(test_scene, 'name')
    assert hasattr(test_scene, 'thumbnail')
    assert hasattr(test_scene, 'main_view_data')
    assert hasattr(test_scene, 'named_view')
    assert hasattr(test_scene, 'scene_models')

from chimerax.core.models import Model
class TestModel(Model):
    """
    A test model class that implements the Scene interface.
    """

    def take_snapshot(self, session, flags):
        return {}

    def restore_snapshot(self, data):
        pass

def test_models_removed(test_production_session):
    """
    Test that the models_removed callback removes models from the scene data. First we test that the callback function
    removes the model data from the scene correctly. Second, we test that the callback function is called when a model
    is removed from the models' manager.
    """
    models_mgr = test_production_session.models
    test_model1 = TestModel('test_model1', test_production_session)
    models_mgr.add([test_model1])

    scenes_mgr = test_production_session.scenes

    # save a scene
    scenes_mgr.save_scene("test_scene")
    test_scene1 = scenes_mgr.get_scene("test_scene")

    # Make sure model1 was added to the scene data
    assert test_model1 in test_scene1.scene_models
    assert test_model1 in test_scene1.named_view.positions

    # Directly test removing model1 from the scene with models_removed
    test_scene1.models_removed([models_mgr[0]])

    # Check that the model1 was removed from the scene data
    assert test_model1 not in test_scene1.scene_models
    assert test_model1 not in test_scene1.named_view.positions

    # Remove the model that we removed from the scene from the session
    models_mgr.remove([test_model1])

    # Create a new model and add it to the scene
    test_model2 = TestModel('test_model2', test_production_session)
    models_mgr.add([test_model2])

    # save a scene
    scenes_mgr.save_scene("test_scene2")
    test_scene2 = scenes_mgr.get_scene("test_scene2")

    # Make sure that the model2 was added to the scene data
    assert test_model2 in test_scene2.scene_models
    assert test_model2 in test_scene2.named_view.positions

    # Remove a model with the models' manager. This should trigger the models_removed callback in the scene
    models_mgr.remove([test_model2])

    # Check that the model2 was removed from the scene data
    assert test_model2 not in test_scene2.scene_models
    assert test_model2 not in test_scene2.named_view.positions

def test_get_name(test_production_session):
    scenes_mgr = test_production_session.scenes
    scenes_mgr.save_scene("test_scene")

    test_scene = scenes_mgr.get_scene("test_scene")
    assert test_scene.get_name() == "test_scene"

def test_restore_snapshot(test_production_session):
    """
    Test that restore_snapshot restores the scene data from the snapshot data with the correct version number.
    """
    scenes_mgr = test_production_session.scenes
    scenes_mgr.save_scene("test_scene")

    test_scene = scenes_mgr.get_scene("test_scene")
    snapshot_data = test_scene.take_snapshot(test_production_session, 0)
    snapshot_data['version'] = -1

    try:
        test_scene.restore_snapshot(test_production_session, snapshot_data)
    except ValueError as e:
        assert str(e) == "Cannot restore Scene data with version -1"
    else:
        assert False, "ValueError not raised when restoring invalid scene version number"
