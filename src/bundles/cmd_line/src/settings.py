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


from  chimerax.core.settings import Settings

class _CmdLineSettings(Settings):
    EXPLICIT_SAVE = {
        'startup_commands': [],
        'default_side': 'bottom'
    }
    AUTO_SAVE = {
        "num_remembered": 2000,
        "typed_only": True,
        "select_failed": False,
    }

# 'settings' module attribute will be set by the initialization of the bundle API

def register_settings_options(session):
    from chimerax.ui.options import StringsOption, EnumOption
    class CmdLinePlacementOption(EnumOption):
        values = ("left", "right", "bottom")

    settings_info = {
        'startup_commands': (
            "Execute these commands at startup",
            StringsOption,
            "List of commands to execute when ChimeraX command-line tool starts"
        ),
    }
    for setting, setting_info in settings_info.items():
        opt_name, opt_class, balloon = setting_info
        opt = opt_class(opt_name, getattr(settings, setting), None,
            attr_name=setting, settings=settings, balloon=balloon)
        session.ui.main_window.add_settings_option("Startup", opt)
    settings_info = {
        'default_side': (
            "Default command line location",
            CmdLinePlacementOption,
            "Where to place the command line by default on startup"
        )
    }
    for setting, setting_info in settings_info.items():
        opt_name, opt_class, balloon = setting_info
        opt = opt_class(opt_name, getattr(settings, setting), None,
            attr_name=setting, settings=settings, balloon=balloon)
        session.ui.main_window.add_settings_option("Window", opt)
