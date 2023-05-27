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

from .gaussian import gaussian_convolve
from .laplace import laplacian
from .fourier import fourier_transform
from .median import median_filter
from .permute import permute_axes
from .zone import zone_volume

# -----------------------------------------------------------------------------
#
from chimerax.core.toolshed import BundleAPI

class _MapFilterBundle(BundleAPI):

    @staticmethod
    def start_tool(session, tool_name):
        if tool_name == 'Map Filter':
            from .filtergui import show_map_filter_panel
            return show_map_filter_panel(session)

    @staticmethod
    def register_command(command_name, logger):
        from .vopcommand import register_volume_filtering_subcommands
        register_volume_filtering_subcommands(logger)

bundle_api = _MapFilterBundle()
