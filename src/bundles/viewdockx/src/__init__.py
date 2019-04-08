# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


class _MyAPI(BundleAPI):

    api_version = 1

    @staticmethod
    def start_tool(session, bi, ti):
        if ti.name == "ViewDockX":
            from .tool import TableTool
            from chimerax.core.errors import UserError
            try:
                tool = TableTool(session, ti.name)
                tool.setup()
            except UserError as e:
                session.logger.error(str(e))
                return None
            return tool
        else:
            raise ValueError("trying to start unknown tool: %s" % ti.name)

    @staticmethod
    def register_command(bi, ci, logger):
        if ci.name == "viewdockx":
            from . import cmd
            func = cmd.viewdock
            desc = cmd.viewdock_desc
        else:
            raise ValueError("trying to register unknown command: %s" % ci.name)
        if desc.synopsis is None:
            desc.synopsis = ci.synopsis
        from chimerax.core.commands import register
        register(ci.name, desc, func)

    @staticmethod
    def open_file(session, stream, file_name, auto_style=True, atomic=True):
        from .io import open_mol2
        return open_mol2(session, stream, file_name, auto_style, atomic)


    @staticmethod
    def get_class(class_name):
        if class_name in ["TableTool", "ChartTool", "PlotTool"]:
            from . import tool
            return getattr(tool, class_name, None)
        else:
            return None


bundle_api = _MyAPI()


import traceback
print("viewdockx imported")
traceback.print_stack()
