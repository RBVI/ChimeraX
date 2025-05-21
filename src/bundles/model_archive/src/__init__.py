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

# -----------------------------------------------------------------------------
#
from chimerax.core.toolshed import BundleAPI

class _ModelArchiveBundle(BundleAPI):
            
    @staticmethod
    def register_command(command_name, logger):
        # 'register_command' is lazily called when the command is referenced
        if command_name == 'modelcif pae':
            from . import modelcif_pae
            modelcif_pae.register_command(logger)

    @staticmethod
    def run_provider(session, name, mgr):
        if mgr == session.open_command:
            if name == 'modelarchive':
                from chimerax.open_command import FetcherInfo
                class MAInfo(FetcherInfo):
                    def fetch(self, session, ident, format_name, ignore_cache, **kw):
                        from .ma_fetch import fetch_model_archive
                        return fetch_model_archive(session, ident, ignore_cache=ignore_cache, **kw)
                    @property
                    def fetch_args(self):
                        from chimerax.core.commands import EnumOf, BoolArg
                        return {
                            'pae': BoolArg,
                        }
                return MAInfo()

bundle_api = _ModelArchiveBundle()
