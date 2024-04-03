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

# ensure C++ shared libs are linkable by us
import chimerax.atomic_lib  # noqa
import chimerax.pdb_lib  # noqa


from chimerax.core.toolshed import BundleAPI

class _MyAPI(BundleAPI):

    # @staticmethod
    # def get_class(class_name):
    #     # 'get_class' is called by session code to get class saved in a session
    #     return None

    @staticmethod
    def run_provider(session, name, mgr):
        if name == "Macromolecular Transmission":
            from chimerax.open_command import OpenerInfo
            class Info(OpenerInfo):
                def open(self, session, data, file_name, **kw):
                    from . import mmtf
                    return mmtf.open_mmtf(session, data, file_name, **kw)

                @property
                def open_args(self):
                    from chimerax.core.commands import BoolArg
                    return {
                        'auto_style': BoolArg,
                        'coordsets': BoolArg
                    }
        else:
            from chimerax.open_command import FetcherInfo
            class Info(FetcherInfo):
                def fetch(self, session, ident, format_name, ignore_cache, **kw):
                    from . import mmtf
                    return mmtf.fetch_mmtf(session, ident, ignore_cache, **kw)
        return Info()

bundle_api = _MyAPI()
