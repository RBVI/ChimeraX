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

class StdCommandsAPI(BundleAPI):

    @staticmethod
    def register_command(command_name, logger):
        # 'register_command' is lazily called when the command is referenced
        primary_name, remainder = command_name.split(None, 1)
        if primary_name[0] == '~':
            primary_name = primary_name[1:]
            # ~show/~display need to map to a different module than show/display
            if primary_name in ("show", "display"):
                primary_name = "hide"
        name_remapping = {
            'colordef': 'colorname',
            'color delete': 'colorname',
            'color list': 'colorname',
            'color name': 'colorname',
            'color show': 'colorname',
            'redo': 'undo'
        }
        primary_name = name_remapping.get(primary_name, primary_name).replace(' ', '_')
        from importlib import import_module
        mod = import_module(".%s" % primary_name, __package__)
        mod.register_command(logger)

    @staticmethod
    def register_selector(selector_name, logger):
        # 'register_selector' is lazily called when selector is referenced
        from .selectors import register_selectors
        register_selectors(logger)

bundle_api = StdCommandsAPI()
