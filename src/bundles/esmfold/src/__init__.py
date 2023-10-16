# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.toolshed import BundleAPI

class _ESMFoldBundle(BundleAPI):

    @staticmethod
    def start_tool(session, tool_name):
        if tool_name == 'ESMFold':
            from . import panel
            return panel.show_esmfold_panel(session)
        elif tool_name == 'ESMFold Coloring':
            from . import panel
            return panel.show_esmfold_coloring_panel(session)
        elif tool_name == 'ESMFold Error Plot':
            from . import pae
            return pae.show_esmfold_error_plot_panel(session)

    @staticmethod
    def register_command(command_name, logger):
        if command_name == 'esmfold contacts':
            from . import contacts
            contacts.register_esmfold_contacts_command(logger)
        elif command_name == 'esmfold fetch':
            from . import fetch
            fetch.register_esmfold_fetch_command(logger)
        elif command_name == 'esmfold match':
            from . import match
            match.register_esmfold_match_command(logger)
        elif command_name == 'esmfold predict':
            from . import predict
            predict.register_esmfold_predict_command(logger)
        elif command_name == 'esmfold pae':
            from . import pae
            pae.register_esmfold_pae_command(logger)
        elif command_name == 'esmfold search':
            from . import blast
            blast.register_esmfold_search_command(logger)

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
