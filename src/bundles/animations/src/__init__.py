__version__ = 1.0

from chimerax.core.toolshed import BundleAPI

class _MyAPI(BundleAPI):
    api_version = 1

    # Override method
    @staticmethod
    def start_tool(session, bi, ti):
        if ti.name == "Animations":
            from .tool import AnimationsTool
            return AnimationsTool(session, ti.name)
        raise ValueError("trying to start unknown tool: %s" % ti.name)

    @staticmethod
    def get_class(class_name):
        # class_name will be a string
        if class_name == "AnimationsTool":
            from .tool import AnimationsTool
            return AnimationsTool
        elif class_name == "Animation":
            from .animation import Animation
            return Animation
        elif class_name == "SceneAnimation":
            from .scene_animation import SceneAnimation
            return SceneAnimation
        raise ValueError("Unknown class name '%s'" % class_name)

    @staticmethod
    def initialize(session, bundle_info):
        """Register animation state managers into the session."""
        from .animation import Animation
        session.add_state_manager("animations", Animation(session))

        from .scene_animation import SceneAnimation
        session.add_state_manager("scene animations", SceneAnimation(session))

    @staticmethod
    def register_command(bi, ci, logger):
        from . import cmd
        cmd.register_command(ci.name, logger)


# Create the ``bundle_api`` object that ChimeraX expects.
bundle_api = _MyAPI()
