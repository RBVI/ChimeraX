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

def test_take_thumbnail(*args):
    """
    Use this function to patch the Scene class take_thumbnail method because it requires a GUI to run and this test is
    running in no GUI mode.
    """
    return 'base64_encoded_thumbnail_string'

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