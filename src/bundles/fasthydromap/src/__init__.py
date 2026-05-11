# === UCSF ChimeraX Copyright ===
# Copyright 2026 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.toolshed import BundleAPI

__version__ = "0.1"


class _FastHydroMapAPI(BundleAPI):

    @staticmethod
    def register_command(command_name, logger):
        if command_name == "fasthydromap":
            from .cmd import register_fasthydromap_command
            register_fasthydromap_command(logger)
        elif command_name == "fasthydromap install":
            from .install import register_fasthydromap_install_command
            register_fasthydromap_install_command(logger)


bundle_api = _FastHydroMapAPI()
