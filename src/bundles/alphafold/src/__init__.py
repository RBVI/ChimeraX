# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.toolshed import BundleAPI

class _AlphaFoldBundle(BundleAPI):

    @staticmethod
    def start_tool(session, tool_name):
        from . import panel
        return panel.show_alphafold_panel(session)

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

    @staticmethod
    def run_provider(session, name, mgr):
        if mgr == session.open_command:
            from chimerax.open_command import FetcherInfo
            class Info(FetcherInfo):
                def fetch(self, session, ident, format_name, ignore_cache, **kw):
                    from .fetch import alphafold_fetch
                    return alphafold_fetch(session, ident, ignore_cache=ignore_cache,
                                           add_to_session=False, **kw)
                @property
                def fetch_args(self):
                    from chimerax.core.commands import BoolArg, Or, EnumOf
                    from chimerax.atomic import ChainArg
                    return {
                        'color_confidence': BoolArg,
                        'align_to': ChainArg,
                        'trim': BoolArg,
                    }
            return Info()

bundle_api = _AlphaFoldBundle()
