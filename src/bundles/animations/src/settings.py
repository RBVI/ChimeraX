# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2025 Regents of the University of California. All rights reserved.
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

class _AnimationsSettings(Settings):
    EXPLICIT_SAVE = {
        'recording_resolution': '1080p',  # Default to 1080p
        'animation_mode': 'scene',  # Default to scene mode ('keyframe' or 'scene')
    }
    AUTO_SAVE = {}

# Global settings instance
_settings = None

def get_settings(session):
    """Get the animations settings instance"""
    global _settings
    if _settings is None:
        _settings = _AnimationsSettings(session, "animations")
    return _settings

def register_settings_options(session):
    """Register animations settings in ChimeraX main settings menu"""
    from chimerax.ui.options import SymbolicEnumOption

    class RecordingResolutionOption(SymbolicEnumOption):
        values = ("1080p", "4k", "custom")
        labels = ("1080p (1920x1080)", "4K UHD (3840x2160)", "Custom Resolution")

    class AnimationModeOption(SymbolicEnumOption):
        values = ("keyframe", "scene")
        labels = ("Keyframe Mode", "Scene Mode")

    settings = get_settings(session)

    settings_info = {
        'recording_resolution': (
            "Default recording resolution",
            RecordingResolutionOption,
            "Default resolution for recording animations to video files"
        ),
        'animation_mode': (
            "Animation tool mode",
            AnimationModeOption,
            "Default mode for the Animation tool (Keyframe or Scene)"
        ),
    }

    for setting, setting_info in settings_info.items():
        opt_name, opt_class, balloon = setting_info
        if isinstance(opt_class, tuple):
            opt_class, kw = opt_class
        else:
            kw = {}
        opt = opt_class(opt_name, getattr(settings, setting), None,
            attr_name=setting, settings=settings, balloon=balloon, **kw)
        session.ui.main_window.add_settings_option("Animations", opt)
