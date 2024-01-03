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
            def open(self, session, data, file_name, *, _name=name, show_tool=True, **kw):
                if _name == "AutoDock PDBQT":
                    from .pdbqt import open_pdbqt
                    opener = open_pdbqt
                elif "Mol2" in name:
                    from .io import open_mol2
                    opener = open_mol2
                elif _name == "SwissDock":
                    from .io import open_swissdock
                    opener = open_swissdock
                else: # ZDOCK
                    from .io import open_zdock
                    opener = open_zdock
                # the below code is also in the Maestro bundle
                models, status = opener(session, data, file_name, True, True)
                all_models = sum([m.all_models() for m in models], start=[])
                if show_tool and session.ui.is_gui and len(all_models) > 1:
                    for m in all_models:
                        if hasattr(m, 'viewdockx_data'):
                            show_dock = True
                            break
                    else:
                        show_dock = False
                    if show_dock:
                        from Qt.QtCore import QTimer
                        from chimerax.core.commands import run, concise_model_spec
                        QTimer.singleShot(0,
                            lambda *args, run=run, ses=session, spec=concise_model_spec, models=models:
                                run(ses, "viewdockx %s" % spec(ses, models)))
                return models, status

            @property
            def open_args(self):
                from chimerax.core.commands import BoolArg
                return { 'show_tool': BoolArg }

        return ViewDockOpenerInfo()


bundle_api = _MyAPI()
