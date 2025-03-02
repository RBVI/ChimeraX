from chimerax.core.toolshed import BundleAPI

class _MyAPI(BundleAPI):
    api_version = 1

    @staticmethod
    def start_tool(session, bi, ti):
        if ti.name == "ViewDock":
            from .tool import ViewDockTool
            tool = ViewDockTool(session, ti.name)
            return tool

    @staticmethod
    def register_command(bi, ci, logger):
        from . import cmd
        cmd.register_command(ci)

    @staticmethod
    def get_class(name):
        if name == "ViewDockTool":
            from .tool import ViewDockTool
            return ViewDockTool
        return None