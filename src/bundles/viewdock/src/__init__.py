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

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        from chimerax.open_command import OpenerInfo
        class ViewDockOpenerInfo(OpenerInfo):
            def open(self, session, data, file_name, *, _name=name, show_tool=True, **kw):
                if _name == "vd_AutoDock PDBQT":
                    from .pdbqt import open_pdbqt
                    opener = open_pdbqt
                elif "Mol2" in name:
                    from .io import open_mol2
                    opener = open_mol2
                elif _name == "vd_SwissDock":
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
                        QTimer.singleShot(0, lambda s=session, m=models: open_viewdock_tool(s, m))
                return models, status

            @property
            def open_args(self):
                from chimerax.core.commands import BoolArg
                return { 'show_tool': BoolArg }

        return ViewDockOpenerInfo()

bundle_api = _MyAPI()

def open_viewdock_tool(session, structures=None):
    """
    Open the ViewDock tool for the given structures. If no structures are given, the tool will handle finding them.

    Args:
        session: the ChimeraX session
        structures: a list of structures to open in the tool
    """
    toolshed = session.toolshed

    bi = toolshed.find_bundle("ViewDock", session.logger)
    bi.start_tool(session, "ViewDock")

