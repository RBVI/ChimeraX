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

"""
preferences: manage preferences
===============================

TODO
"""
from . import commands
from .colors import Color
from . import configfile
from .settings import Settings


class _CoreSettings(Settings):

    EXPLICIT_SAVE = {
        'atomspec_contents': 'simple', # choices: simple, command-line specifier, serial number
        'bg_color': configfile.Value(
            Color('#000'), commands.ColorArg, Color.hex_with_alpha),
        'multisample_threshold': configfile.Value(
            0, commands.NonNegativeIntArg, str),
        'silhouette': False,
    }

def init(session):
    global settings
    settings = _CoreSettings(session, "chimerax.core")
