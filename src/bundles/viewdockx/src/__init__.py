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
        from . import cmd
        cmd.register_command(ci)

    @staticmethod
    def open_file(session, path, file_name, format_name,
                  auto_style=True, atomic=True):
        if format_name == "mol2":
            from .io import open_mol2
            return open_mol2(session, path, file_name, auto_style, atomic)
        elif format_name == "pdbqt":
            from .pdbqt import open_pdbqt
            return open_pdbqt(session, path, file_name, auto_style, atomic)
        else:
            raise ValueError("unsupported format: %s" % format_name)


    @staticmethod
    def get_class(class_name):
        if class_name in ["TableTool", "ChartTool", "PlotTool"]:
            from . import tool
            return getattr(tool, class_name, None)
        else:
            return None


bundle_api = _MyAPI()
