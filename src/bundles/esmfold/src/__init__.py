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

class _ESMFoldBundle(BundleAPI):

    @staticmethod
    def register_command(command_name, logger):
        if command_name == 'esmfold fetch':
            from . import fetch
            fetch.register_esmfold_fetch_command(logger)
        elif command_name == 'esmfold predict':
            from . import predict
            predict.register_esmfold_predict_command(logger)
        elif command_name == 'esmfold pae':
            from . import pae
            pae.register_esmfold_pae_command(logger)

    @staticmethod
    def run_provider(session, name, mgr):
        if mgr == session.open_command:
            from chimerax.open_command import FetcherInfo
            class Info(FetcherInfo):
                def fetch(self, session, ident, format_name, ignore_cache, **kw):
                    from .fetch import esmfold_fetch
                    return esmfold_fetch(session, ident, ignore_cache=ignore_cache,
                                         add_to_session=False, in_file_history=False,
                                         **kw)
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

bundle_api = _ESMFoldBundle()
