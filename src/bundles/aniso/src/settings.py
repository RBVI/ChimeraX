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

defaults = {
    "presets": {
        "simple ellipsoid": {
            'intrinsic': True,
            'axis_color': None,
            'axis_factor': None,
            'axis_thickness': 0.01,
            'ellipse_color': None,
            'ellipse_factor': None,
            'ellipse_thickness': 0.02,
            'ellipsoid_color': None,
            'scale': 1.0,
            'show_ellipsoid': True,
            'smoothing': 3,
            'transparency': None,
        },
        "principal axes": {
            'intrinsic': True,
            'axis_color': None,
            'axis_factor': 1.0,
            'axis_thickness': 0.01,
            'ellipse_color': None,
            'ellipse_factor': None,
            'ellipse_thickness': 0.02,
            'ellipsoid_color': None,
            'scale': 1.0,
            'show_ellipsoid': False,
            'smoothing': 3,
            'transparency': None,
        },
        "principal ellipses": {
            'intrinsic': True,
            'axis_color': None,
            'axis_factor': None,
            'axis_thickness': 0.01,
            'ellipse_color': None,
            'ellipse_factor': 1.0,
            'ellipse_thickness': 0.02,
            'ellipsoid_color': None,
            'scale': 1.0,
            'show_ellipsoid': False,
            'smoothing': 3,
            'transparency': None,
        },
        "ellipsoid and principal axes": {
            'intrinsic': True,
            'axis_color': None,
            'axis_factor': 1.5,
            'axis_thickness': 0.01,
            'ellipse_color': None,
            'ellipse_factor': None,
            'ellipse_thickness': 0.02,
            'ellipsoid_color': None,
            'scale': 1.0,
            'show_ellipsoid': True,
            'smoothing': 3,
            'transparency': None,
        },
        "octant lines": {
            'intrinsic': True,
            'axis_color': None,
            'axis_factor': None,
            'axis_thickness': 0.01,
            'ellipse_color': (0,0,0,255),
            'ellipse_factor': 1.01,
            'ellipse_thickness': 0.02,
            'ellipsoid_color': None,
            'scale': 1.0,
            'show_ellipsoid': True,
            'smoothing': 3,
            'transparency': None,
        },
        "snow globe axes": {
            'intrinsic': True,
            'axis_color': None,
            'axis_factor': 0.99,
            'axis_thickness': 0.01,
            'ellipse_color': None,
            'ellipse_factor': None,
            'ellipse_thickness': 0.02,
            'ellipsoid_color': (255,255,255,255),
            'scale': 1.0,
            'show_ellipsoid': True,
            'smoothing': 3,
            'transparency': 50,
        },
        "snow globe ellipses": {
            'intrinsic': True,
            'axis_color': None,
            'axis_factor': None,
            'axis_thickness': 0.01,
            'ellipse_color': None,
            'ellipse_factor': 0.99,
            'ellipse_thickness': 0.02,
            'ellipsoid_color': (255,255,255,255),
            'scale': 1.0,
            'show_ellipsoid': True,
            'smoothing': 3,
            'transparency': 50,
        },
    },
}

from  chimerax.core.settings import Settings
from copy import deepcopy

class _AnisoSettings(Settings):
	EXPLICIT_SAVE = deepcopy(defaults)

_settings = None
def get_settings(session):
    global _settings
    if _settings is None:
        _settings = _AnisoSettings(session, "Thermal Ellipsoids")
    return _settings
