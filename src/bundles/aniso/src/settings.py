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
from chimerax.core.configfile import Value
import json

# don't include built-in presets in the settings like I was doing earlier 
# (with an 'intrinsic' attribute) because then once the user creates a
# custom setting, the built-in ones will be sitting in the user's settings
# and cannot be altered or added to by later versions of ChimeraX
class AnisoSettings(Settings):
    AUTO_SAVE = {
        "custom_presets": Value({}, json.loads, json.dumps),
    }


_settings = None
def get_settings(session):
    global _settings
    if _settings is None:
        _settings = AnisoSettings(session, "Thermal Ellipsoids")
    return _settings
