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

class _PhenixBundle(BundleAPI):

    @staticmethod
    def start_tool(session, tool_name):
        if tool_name == 'Water Placement':
            from . import douse
            return douse.show_water_placement_tool(session)

    @staticmethod
    def register_command(command_name, logger):
        if command_name == 'phenix douse':
            from . import douse
            douse.register_phenix_douse_command(logger)
        elif command_name == 'phenix location':
            from . import locate
            locate.register_phenix_location_command(logger)

bundle_api = _PhenixBundle()
