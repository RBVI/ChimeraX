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

from .fetch_uniprot import map_uniprot_ident

from chimerax.core.toolshed import BundleAPI

class _UniprotBundleAPI(BundleAPI):

    @staticmethod
    def run_provider(session, name, mgr, *, widget_info=None, **kw):
        from chimerax.open_command import FetcherInfo
        class UniprotFetcherInfo(FetcherInfo):
            def fetch(self, session, ident, format_name, ignore_cache, *, associate=None, **kw):
                from .fetch_uniprot import fetch_uniprot
                return fetch_uniprot(session, ident, ignore_cache=ignore_cache, associate=associate)

            @property
            def fetch_args(self):
                from chimerax.atomic import UniqueChainsArg
                return { 'associate': UniqueChainsArg }

        return UniprotFetcherInfo()

bundle_api = _UniprotBundleAPI()
