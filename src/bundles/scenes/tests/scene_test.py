from chimerax.scenes import scene
from chimerax.core.models import Model



class MockModelA:
    def restore_scene(self, data):
        pass

class MockModelB(MockModelA):
    pass

class MockModelC(MockModelB):
    def restore_scene(self, data):
        pass

class MockModelD:
    pass

def test_md_scene_implementation():
    """
    Test that md_scene_implementation returns the most derived class that implements restore_scene.
    """
    model_a = MockModelA()
    model_b = MockModelB()
    model_c = MockModelC()

    # MockModelA implements restore_scene
    assert scene.md_scene_implementation(model_a) == MockModelA
    # MockModelB does not implement restore_scene, but it's super class does
    assert scene.md_scene_implementation(model_b) == MockModelA
    # MockModelC implements restore_scene
    assert scene.md_scene_implementation(model_c) == MockModelC
    # MockModelD does not implement restore_scene and neither do any of its super classes
    assert scene.md_scene_implementation(MockModelD) is None

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
