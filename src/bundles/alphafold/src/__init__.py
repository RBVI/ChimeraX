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

class _AlphaFoldBundle(BundleAPI):

    @staticmethod
    def start_tool(session, tool_name):
        if tool_name == 'AlphaFold':
            from . import panel
            return panel.show_alphafold_panel(session)
        elif tool_name == 'AlphaFold Coloring':
            from . import colorgui
            return colorgui.show_alphafold_coloring_panel(session)
        elif tool_name == 'AlphaFold Error Plot':
            from . import pae
            return pae.show_alphafold_error_plot_panel(session)

    @staticmethod
    def register_command(command_name, logger):
        if command_name == 'alphafold match':
            from . import match
            match.register_alphafold_match_command(logger)
        elif command_name == 'alphafold fetch':
            from . import fetch
            fetch.register_alphafold_fetch_command(logger)
        elif command_name == 'alphafold search':
            from . import blast
            blast.register_alphafold_search_command(logger)
        elif command_name == 'alphafold predict':
            from . import predict
            predict.register_alphafold_predict_command(logger)
        elif command_name == 'alphafold pae':
            from . import pae
            pae.register_alphafold_pae_command(logger)
        elif command_name == 'alphafold contacts':
            from . import contacts
            contacts.register_alphafold_contacts_command(logger)
        elif command_name == 'alphafold covariation':
            from . import msa
            msa.register_alphafold_covariation_command(logger)
        elif command_name == 'alphafold msa':
            from . import msa
            msa.register_alphafold_msa_command(logger)
        elif command_name == 'alphafold dimers':
            from . import dimers
            dimers.register_alphafold_dimers_command(logger)
        elif command_name == 'alphafold monomers':
            from . import dimers
            dimers.register_alphafold_monomers_command(logger)
        elif command_name == 'alphafold interfaces':
            from . import interfaces
            interfaces.register_alphafold_interfaces_command(logger)

    @staticmethod
    def run_provider(session, name, mgr):
        if mgr == session.open_command:
            if name == 'alphafold':
                from chimerax.open_command import FetcherInfo
                class AlphaFoldDatabaseInfo(FetcherInfo):
                    def fetch(self, session, ident, format_name, ignore_cache, **kw):
                        from .fetch import alphafold_fetch
                        return alphafold_fetch(session, ident, ignore_cache=ignore_cache,
                                               add_to_session=False, in_file_history=False,
                                               **kw)
                    @property
                    def fetch_args(self):
                        from chimerax.core.commands import BoolArg, Or, EnumOf, IntArg
                        from chimerax.atomic import ChainArg
                        return {
                            'color_confidence': BoolArg,
                            'align_to': ChainArg,
                            'trim': BoolArg,
                            'pae': BoolArg,
                            'version': IntArg,
                        }
                return AlphaFoldDatabaseInfo()
            elif name == 'alphafold_pae':
                from chimerax.open_command import FetcherInfo
                class AlphaFoldDatabasePAEInfo(FetcherInfo):
                    def fetch(self, session, ident, format_name, ignore_cache, **kw):
                        if 'structure' not in kw:
                            from chimerax.core.errors import UserError
                            raise UserError('Fetching an AlphaFold database PAE file requires specifying the structure to associate.  For example "open P29474 from alphafold_pae structure #1"')
                        from . import pae
                        pae.alphafold_pae(session, uniprot_id = ident, ignore_cache = ignore_cache, **kw)
                        return [], f'Opened AlphaFold database PAE for UniProt id {ident}'
                    @property
                    def fetch_args(self):
                        from chimerax.core.commands import BoolArg, FloatArg, IntArg
                        from chimerax.atomic import AtomicStructureArg
                        return {
                            'structure': AtomicStructureArg,
                            'plot': BoolArg,
                            'divider_lines': BoolArg,
                            'color_domains': BoolArg,
                            'connext_max_pae': FloatArg,
                            'cluster': FloatArg,
                            'min_size': IntArg,
                        }
                return AlphaFoldDatabasePAEInfo()
            elif name == 'AlphaFold PAE':
                from chimerax.open_command import OpenerInfo
                class AlphaFoldPAEInfo(OpenerInfo):
                    def open(self, session, path, file_name, **kw):
                        from .pae import alphafold_pae
                        pae = alphafold_pae(session, file = path, **kw)
                        return [], f'Opened AlphaFold PAE with values for {pae.matrix_size} residues and atoms'

                    @property
                    def open_args(self):
                        from chimerax.core.commands import BoolArg, FloatArg, IntArg
                        from chimerax.atomic import AtomicStructureArg
                        return {
                            'structure': AtomicStructureArg,
                            'plot': BoolArg,
                            'divider_lines': BoolArg,
                            'color_domains': BoolArg,
                            'connext_max_pae': FloatArg,
                            'cluster': FloatArg,
                            'min_size': IntArg,
                        }

                return AlphaFoldPAEInfo()
            
    @staticmethod
    def get_class(class_name):
        # 'get_class' is called by session code to get class saved in a session
        if class_name == 'DatabaseEntryId':
            from .search import DatabaseEntryId
            return DatabaseEntryId

bundle_api = _AlphaFoldBundle()
