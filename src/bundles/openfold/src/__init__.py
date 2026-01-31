# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
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

class _OpenFoldBundle(BundleAPI):

    @staticmethod
    def start_tool(session, tool_name):
        if tool_name == 'OpenFold':
            from . import openfold_gui
            return openfold_gui.show_openfold_panel(session)
        elif tool_name == 'OpenFold History':
            from . import history
            return history.show_predictions_panel(session)

    @staticmethod
    def register_command(command_name, logger):
        if command_name in ('openfold predict', 'openfold ligandtable'):
            from . import predict
            predict.register_openfold_predict_command(logger)
        elif command_name == 'openfold install':
            from . import install
            install.register_openfold_install_command(logger)

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        if mgr == session.open_command:
            from chimerax.open_command import OpenerInfo
            if name == 'OpenFold ligands':
                class OpenFoldLigandsInfo(OpenerInfo):
                    def open(self, session, path, file_name, **kw):
                        if session.ui.is_gui:
                            from . import openfold_gui
                            olt = openfold_gui.open_openfold_ligands_file(session, path, align_to = kw.get('align_to'))
                            msg = f'Read {olt.ligand_count()} ligands'
                        else:
                            msg = 'Cannot show ligands table in no-GUI mode.'
                        return [], msg
                    @property
                    def open_args(self):
                        from chimerax.atomic import AtomicStructureArg
                        return { 'align_to': AtomicStructureArg, }
                return OpenFoldLigandsInfo()

    # Make class name to class for session restore
    @staticmethod
    def get_class(class_name):
        if class_name == 'OpenFoldHistoryPanel':
            from .history import OpenFoldHistoryPanel
            return OpenFoldHistoryPanel

bundle_api = _OpenFoldBundle()
