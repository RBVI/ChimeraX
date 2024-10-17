# vim: set expandtab ts=4 sw=4:

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

class _DeepMutationalScanAPI(BundleAPI):

    @staticmethod
    def register_command(command_name, logger):
        # 'register_command' is called by the toolshed on start up
        from . import ms_data
        ms_data.register_commands(logger)
        from . import ms_label
        ms_label.register_command(logger)
        from . import ms_define
        ms_define.register_command(logger)
        from . import ms_scatter_plot
        ms_scatter_plot.register_command(logger)
        from . import ms_stats
        ms_stats.register_command(logger)
        from . import ms_histogram
        ms_histogram.register_command(logger)
        from . import ms_umap
        ms_umap.register_command(logger)

    @staticmethod
    def run_provider(session, name, mgr):
        if mgr == session.open_command:
            if name == 'Deep mutational scan':
                from chimerax.open_command import OpenerInfo
                class DeepMutationalScanInfo(OpenerInfo):
                    def open(self, session, path, file_name, **kw):
                        from .ms_data import open_deep_mutational_scan_csv
                        dms_data, message = open_deep_mutational_scan_csv(session, path, **kw)
                        return [], message

                    @property
                    def open_args(self):
                        from chimerax.atomic import ChainArg
                        return {'chain': ChainArg}

                return DeepMutationalScanInfo()

bundle_api = _DeepMutationalScanAPI()
