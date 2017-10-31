# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


# Subclass from chimerax.core.toolshed.BundleAPI and
# override the method for registering commands,
# inheriting all other methods from the base class.
class _MyAPI(BundleAPI):

    api_version = 1     # register_command called with BundleInfo and
                        # CommandInfo instance instead of command name
                        # (when api_version==0)

    # Override method
    @staticmethod
    def start_tool(bi, ti, logger):
        # bi is an instance of chimerax.core.toolshed.BundleInfo
        # ti is an instance of chimerax.core.toolshed.ToolInfo
        # logger is an instance of chimerax.core.logger.Logger

        # This method is called once for each time the tool is invoked.

        # We check the name of the tool, which should match one of the
        # ones listed in bundle_info.xml (without the leading and
        # trailing whitespace), and create an instance of the
        # appropriate class from the ``gui`` module.
        from . import gui
        if ti.name == "Tutorial GUI":
            func = gui.TutorialGUI(session, ti)
        else:
            raise ValueError("trying to start unknown tool: %s" % ti.name)


# Create the ``bundle_api`` object that ChimeraX expects.
bundle_api = _MyAPI()
