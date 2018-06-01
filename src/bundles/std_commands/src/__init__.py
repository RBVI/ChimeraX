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
        tilde = command_name[0] == '~'
        check_name = command_name[1:] if tilde else command_name
        name_remapping = {
            'colordef': 'colorname',
            'colourdef': 'colorname',
            'color delete': 'colorname',
            'color list': 'colorname',
            'color name': 'colorname',
            'color show': 'colorname',
            'quit': 'exit',
            'redo': 'undo',
            'select zone': 'zonesel'
        }
        if tilde:
            name_remapping['show'] = name_remapping['display'] = 'hide'
        else:
            name_remapping['display'] = 'show'
        if check_name in name_remapping:
            mod_name = name_remapping[check_name]
        elif check_name.startswith('measure '):
            mod_name = check_name.replace(' ', '_')
        else:
            if ' ' in check_name:
                mod_name, remainder = check_name.split(None, 1)
            else:
                mod_name = check_name
            if mod_name.startswith("colour"):
                mod_name = "color" + mod_name[6:]
        from importlib import import_module
        mod = import_module(".%s" % mod_name, __package__)
        mod.register_command(logger)

    @staticmethod
    def register_selector(selector_name, logger):
        # 'register_selector' is lazily called when selector is referenced
        from .selectors import register_selectors
        register_selectors(logger)

bundle_api = StdCommandsAPI()
