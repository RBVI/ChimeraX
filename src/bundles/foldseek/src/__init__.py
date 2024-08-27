# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
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

class _FoldseekBundle(BundleAPI):

    @staticmethod
    def start_tool(session, tool_name):
        if tool_name == 'Foldseek':
            from . import gui
            return gui.show_foldseek_panel(session)

    @staticmethod
    def register_command(command_name, logger):
        if command_name == 'foldseek':
            from . import search
            search.register_foldseek_command(logger)
        elif command_name in ('foldseek open', 'foldseek pairing', 'foldseek seqalign'):
            from . import foldseek
            foldseek.register_foldseek_command(logger)
        elif command_name == 'foldseek sequences':
            from . import sequences
            sequences.register_foldseek_sequences_command(logger)
        elif command_name == 'foldseek traces':
            from . import traces
            traces.register_foldseek_traces_command(logger)
        elif command_name == 'foldseek cluster':
            from . import cluster
            cluster.register_foldseek_cluster_command(logger)
        elif command_name == 'foldseek ligands':
            from . import ligands
            ligands.register_foldseek_ligands_command(logger)
        elif command_name == 'foldseek scrollto':
            from . import gui
            gui.register_foldseek_scrollto_command(logger)

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        if mgr == session.open_command:
            from chimerax.open_command import OpenerInfo
            class FoldseekInfo(OpenerInfo):
                def open(self, session, path, file_name, **kw):
                    from . import foldseek
                    return foldseek.open_foldseek_m8(session, path, query_chain = kw.get('chain'))
                @property
                def open_args(self):
                    from chimerax.core.commands import EnumOf
                    from chimerax.atomic import ChainArg
                    from .search import foldseek_databases
                    return { 'chain': ChainArg,
                             'database': EnumOf(foldseek_databases)}
            return FoldseekInfo()

    # Make class name to class for session restore
    @staticmethod
    def get_class(class_name):
        if class_name == 'FoldseekPanel':
            from .gui import FoldseekPanel
            return FoldseekPanel
        elif class_name == 'FoldseekResults':
            from .foldseek import FoldseekResults
            return FoldseekResults
            

bundle_api = _FoldseekBundle()
