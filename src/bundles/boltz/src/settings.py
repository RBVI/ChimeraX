# vim: set expandtab ts=4 sw=4:

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

# -----------------------------------------------------------------------------
#
from chimerax.core.settings import Settings

class _BoltzSettings(Settings):
    EXPLICIT_SAVE = {
        'boltz_results_location': '~/Desktop/boltz_[name]',
        'boltz_install_location': '',
        'device': 'default',	# default, cpu, or gpu
        'samples': 1,		# Number of predicted structures
        'use_msa_cache': True,
    }

# -----------------------------------------------------------------------------
#
def _boltz_settings(session):
    settings = getattr(session, '_boltz_settings', None)
    if settings is None:
        settings = _BoltzSettings(session, "boltz_database")
        session._boltz_settings = settings
    return settings
