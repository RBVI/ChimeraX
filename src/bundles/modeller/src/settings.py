# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
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

from chimerax.core.settings import Settings
from .loops import ALL_MISSING

class _ModellerComparativeSettings(Settings):

    AUTO_SAVE = {
        'fast': False,
        'het_preserve': False,
        'hydrogens': False,
        'license_key': None,
        'local_execution': False,
        'executable_path': "",
        'multichain': True,
        'num_models': 5,
        'temp_path': "",
        'water_preserve': False
    }


class _ModellerLoopsSettings(Settings):

    AUTO_SAVE = {
        'adjacent_flexible': 1,
        'fast': False,
        'num_models': 5,
        'protocol': "standard",
        'region': ALL_MISSING,
        'temp_path': "",
    }


_comparative_settings = _loops_settings = None
def get_settings(session, settings_type):
    global _loops_settings, _comparative_settings
    if settings_type == "Modeller Comparative" or settings_type == "license":
        if _comparative_settings is None:
            _comparative_settings = _ModellerComparativeSettings(session, "modeller")
        settings = _comparative_settings
    else:
        if _loops_settings is None:
            _loops_settings = _ModellerLoopsSettings(session, "modeller loops")
        settings = _loops_settings
    return settings
