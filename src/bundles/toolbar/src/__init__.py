#
# vim: set expandtab shiftwidth=4 softtabstop=4:
#

from chimerax.core.toolshed import BundleAPI


class _MyAPI(BundleAPI):

    api_version = 1

    # Override method
    @staticmethod
    def start_tool(session, bi, ti):
        from . import tool
        if ti.name == "Toolbar":
            return tool.ToolbarTool(session, ti.name)
        raise ValueError("trying to start unknown tool: %s" % ti.name)


# Create the ``bundle_api`` object that ChimeraX expects.
bundle_api = _MyAPI()
