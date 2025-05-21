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

class _NMRSTAR_API(BundleAPI):
    @staticmethod
    def initialize(session, bundle_info):
        """Register file formats, commands, and database fetch."""
        from . import nmr_selectors
        nmr_selectors._register_distance_constraint_selectors(session.logger)

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        if mgr == session.open_command:
            from chimerax.core.commands import StringArg, Color8Arg, FloatArg, IntArg
            from chimerax.atomic import AtomicStructuresArg
            nmr_star_args = {
                'structures': AtomicStructuresArg,
                'type': StringArg,		# E.g. "NOE" or "hydrogen bond"
                'color': Color8Arg,		# Satisfied constraint color
                'radius': FloatArg,		# Satisfied constraint radius
                'long_color': Color8Arg,	# Long constraint color
                'long_radius': FloatArg,	# Long constraint radius
                'dashes': IntArg,		# Pseudobond dashes, 0 = no dashes
            }
            if name == 'pdb_nmr':
                from chimerax.open_command import FetcherInfo
                class NMRSTARInfo(FetcherInfo):
                    def fetch(self, session, ident, format_name, ignore_cache, **kw):
                        from . import nmrstar
                        return nmrstar.pdb_fetch(session, ident, ignore_cache=ignore_cache, **kw)
                    @property
                    def fetch_args(self):
                        return nmr_star_args

            elif name == 'NMRSTAR':
                from chimerax.open_command import OpenerInfo
                class NMRSTARInfo(OpenerInfo):
                    def open(self, session, path, file_name, **kw):
                        from . import nmrstar
                        return nmrstar.read_nmr_star(session, path, file_name, **kw)
                    @property
                    def open_args(self):
                        return nmr_star_args

        return NMRSTARInfo()
bundle_api = _NMRSTAR_API()
