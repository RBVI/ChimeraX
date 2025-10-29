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

class AnimationsPreferencesDialog:
    """Dialog for animations preferences"""

    def __init__(self, session, parent=None):
        from Qt.QtWidgets import QDialog, QVBoxLayout
        from chimerax.ui.options import SettingsPanel, SymbolicEnumOption

        self.session = session
        self.settings = get_settings(session)

        self.dialog = QDialog(parent)
        self.dialog.setWindowTitle("Animations Preferences")
        self.dialog.setModal(True)

        layout = QVBoxLayout()

        # Create settings panel with standard buttons (Save, Reset, Restore, Help)
        self.panel = SettingsPanel(scrolled=False, help_cb=self._show_help)

        # Recording resolution option
        self.panel.add_option(
            SymbolicEnumOption(
                name="Default recording resolution",
                default=None,
                attr_name="recording_resolution",
                settings=self.settings,
                callback=None,
                labels=["1080p (1920x1080)", "4K UHD (3840x2160)", "Custom Resolution"],
                values=["1080p", "4k", "custom"]
            )
        )

        # Animation mode option
        self.panel.add_option(
            SymbolicEnumOption(
                name="Animation tool mode",
                default=None,
                attr_name="animation_mode",
                settings=self.settings,
                callback=None,
                labels=["Keyframe Mode", "Scene Mode"],
                values=["keyframe", "scene"]
            )
        )

        layout.addWidget(self.panel)
        self.dialog.setLayout(layout)

    def _show_help(self):
        """Show help for animations preferences"""
        from chimerax.core.commands import run
        run(self.session, "help help:user/tools/animations.html")

    def show(self):
        """Show the dialog"""
        return self.dialog.exec()
