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

from .compare import compare_waters

from chimerax.core.toolshed import BundleAPI

class _CheckWatersBundle(BundleAPI):

    @staticmethod
    def get_class(class_name):
        from . import tool
        return getattr(tool, class_name)

    @staticmethod
    def register_command(command_name, logger):
        from . import cmd
        cmd.register_command(logger)

    @staticmethod
    def start_tool(session, tool_name):
        if tool_name == 'Check Waters':
            from .tool import CheckWatersInputTool, CheckWaterViewer
            from chimerax.atomic import all_atomic_structures as aas
            structures = aas(session)
            if len(structures) == 1:
                from chimerax.map import Volume
                vols = [m for m in session.models if isinstance(m, Volume)]
                if len(vols) < 2:
                    vol_arg = None if not vols else vols[0]
                    if vol_arg is not None:
                        from .tool import check_overlap
                        check_overlap(structures[0], vol_arg)
                    return CheckWaterViewer(session, tool_name, structures[0], compare_map=vol_arg)
            return CheckWatersInputTool(session)

bundle_api = _CheckWatersBundle()
