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
    def initialize(session, bundle_info):
        # 'initialize' is called by the toolshed on start up
        if session.ui.is_gui:
            from .mousemode import register_mousemode
            register_mousemode(session)

    @staticmethod
    def register_command(bi, ci, logger):
        from . import cmd
        cmd.register_command(ci)

    @staticmethod
    def get_class(class_name):
        if class_name in ["TableTool", "ChartTool", "PlotTool"]:
            from . import tool
            return getattr(tool, class_name, None)
        else:
            return None

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        from chimerax.open_command import OpenerInfo
        class ViewDockOpenerInfo(OpenerInfo):
            def open(self, session, data, file_name, *, _name=name, **kw):
                if _name == "AutoDock PDBQT":
                    from .pdbqt import open_pdbqt
                    opener = open_pdbqt
                elif "Mol2" in name:
                    from .io import open_mol2
                    opener = open_mol2
                else: # ZDOCK
                    from .io import open_zdock
                    opener = open_zdock
                return opener(session, data, file_name, True, True)
        return ViewDockOpenerInfo()


bundle_api = _MyAPI()
