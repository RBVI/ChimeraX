__version__ = 1.0

from chimerax.core.toolshed import BundleAPI
from chimerax.animations.tool import AnimationsTool

class _MyAPI(BundleAPI):
    api_version = 1

    # Override method
    @staticmethod
    def start_tool(session, bi, ti):
        if ti.name == "Animations":
            return AnimationsTool(session, ti.name)
        raise ValueError("trying to start unknown tool: %s" % ti.name)

    @staticmethod
    def get_class(class_name):
        # class_name will be a string
        if class_name == "AnimationsTool":
            return AnimationsTool
        elif class_name == "Animation":
            from .animation import Animation
            return Animation
        raise ValueError("Unknown class name '%s'" % class_name)

    @staticmethod
    def initialize(session, bundle_info):
        """Install scene manager into existing session"""
        from .animation import Animation
        session.add_state_manager("animations", Animation(session))
        if session.ui.is_gui:
            from . import settings
            session.ui.triggers.add_handler('ready',
                lambda *args, ses=session: settings.register_settings_options(ses))

    @staticmethod
    def register_command(bi, ci, logger):
        from . import cmd
        cmd.register_command(ci.name, logger)


# Create the ``bundle_api`` object that ChimeraX expects.
bundle_api = _MyAPI()
