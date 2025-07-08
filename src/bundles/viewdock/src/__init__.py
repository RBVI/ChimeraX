# === UCSF ChimeraX Copyright ===
# Copyright 2025 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.toolshed import BundleAPI

RATING_KEY = 'rating'
DEFAULT_RATING = 2

class _MyAPI(BundleAPI):
    api_version = 1

    @staticmethod
    def start_tool(session, bi, ti):
        if ti.name == "ViewDock":
            show_docking_file_dialogue(session)

    @staticmethod
    def register_command(bi, ci, logger):
        from . import cmd
        cmd.register_command(ci)

    @staticmethod
    def get_class(name):
        if name == "ViewDockTool" or name == "TableTool":
            from .tool import ViewDockTool
            return ViewDockTool
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
                if show_tool and session.ui.is_gui:
                    for m in all_models:
                        if hasattr(m, 'viewdock_data'):
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

def show_docking_file_dialogue(session):
    """
    Show an open file dialogue specifically for docking results formats.
    """

    docking_formats_names = []
    for data_format in session.data_formats.formats:
        if data_format.category == "Docking results" or data_format.name == "Sybyl Mol2":
            docking_formats_names.append(data_format.name)
    if not docking_formats_names:
        session.logger.warning("No docking results formats found.")
        return
    from chimerax.open_command import show_open_file_dialog
    show_open_file_dialog(session, format_names=docking_formats_names, caption="Choose Docking Results File")


def open_viewdock_tool(session, structures):
    """
    Open the ViewDock tool for the given structures.

    Args:
        session (Session): the ChimeraX session.
        structures (list): A list of structures to open in the tool.
    """

    if not structures:
        session.logger.warning("Cannot open ViewDock without providing docking structures.")
        return
    from .tool import ViewDockTool
    ViewDockTool(session, "ViewDock", structures)
