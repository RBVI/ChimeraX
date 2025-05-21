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

class _SimilarStructuresBundle(BundleAPI):

    @staticmethod
    def start_tool(session, tool_name):
        if tool_name in ('Similar Structures', 'Foldseek'):
            from . import gui
            return gui.show_similar_structures_panel(session)

    @staticmethod
    def register_command(command_name, logger):
        if command_name == 'foldseek':
            from . import foldseek_search
            foldseek_search.register_foldseek_command(logger)
        elif command_name in ('similarstructures open', 'similarstructures close', 'similarstructures list',
                              'similarstructures pairing', 'similarstructures seqalign'):
            from . import simstruct
            simstruct.register_similar_structures_command(logger)
        elif command_name == 'similarstructures sequences':
            from . import sequences
            sequences.register_similar_structures_sequences_command(logger)
        elif command_name == 'similarstructures traces':
            from . import traces
            traces.register_similar_structures_traces_command(logger)
        elif command_name == 'similarstructures cluster':
            from . import cluster
            cluster.register_similar_structures_cluster_command(logger)
        elif command_name == 'similarstructures ligands':
            from . import ligands
            ligands.register_similar_structures_ligands_command(logger)
        elif command_name == 'similarstructures scrollto':
            from . import gui
            gui.register_similar_structures_scrollto_command(logger)
        elif command_name == 'similarstructures fetchcoords':
            from . import coords
            coords.register_fetchcoords_command(logger)
        elif command_name == 'sequence search':
            from . import mmseqs2_search
            mmseqs2_search.register_mmseqs2_search_command(logger)
        elif command_name == 'similarstructures blast':
            from . import blast_search
            blast_search.register_similar_structures_blast_command(logger)
        elif command_name == 'similarstructures fromblast':
            from . import blast_search
            blast_search.register_similar_structures_from_blast_command(logger)

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        if mgr == session.open_command:
            from chimerax.open_command import OpenerInfo
            if name == 'Similar Structures':
                class SimilarStructuresInfo(OpenerInfo):
                    def open(self, session, path, file_name, **kw):
                        from .simstruct import SimilarStructures
                        results = SimilarStructures.read_sms_file(session, path)
                        if session.ui.is_gui and kw.get('show_table', True):
                            from .gui import show_similar_structures_table
                            show_similar_structures_table(session, results)
                        return [], f'Read {results.description}'
                    @property
                    def open_args(self):
                        from chimerax.core.commands import EnumOf, BoolArg
                        return { 'show_table': BoolArg, }
                return SimilarStructuresInfo()
            elif name == 'Foldseek':
                class FoldseekInfo(OpenerInfo):
                    def open(self, session, path, file_name, **kw):
                        from . import foldseek_search
                        return foldseek_search.open_foldseek_m8(session, path, query_chain = kw.get('chain'))
                    @property
                    def open_args(self):
                        from chimerax.core.commands import EnumOf
                        from chimerax.atomic import ChainArg
                        from .foldseek_search import foldseek_databases
                        return { 'chain': ChainArg,
                                 'database': EnumOf(foldseek_databases)}
                return FoldseekInfo()

    # Make class name to class for session restore
    @staticmethod
    def get_class(class_name):
        if class_name == 'SimilarStructuresPanel':
            from .gui import SimilarStructuresPanel
            return SimilarStructuresPanel
        elif class_name == 'SimilarStructures':
            from .simstruct import SimilarStructures
            return SimilarStructures
        elif class_name == 'SimilarStructuresManager':
            from .simstruct import SimilarStructuresManager
            return SimilarStructuresManager
        elif class_name == 'BackboneTraces':
            from .traces import BackboneTraces
            return BackboneTraces
        elif class_name == 'SimilarStructurePlot':
            from .cluster import SimilarStructurePlot
            return SimilarStructurePlot
            

bundle_api = _SimilarStructuresBundle()
