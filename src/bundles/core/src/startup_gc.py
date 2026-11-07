# vim: set expandtab shiftwidth=4 softtabstop=4:
# === UCSF ChimeraX Copyright ===
# Copyright 2026 Regents of the University of California. All rights reserved.
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

"""Keep cyclic collection from repeatedly scanning long-lived startup objects."""

import gc


class StartupGC:
    def __init__(self, *, freeze_allowed=False):
        self._freeze_allowed = freeze_allowed
        self._paused = False
        self._frozen = False

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def pause(self):
        if gc.isenabled():
            gc.disable()
            self._paused = True

    def resume(self):
        """Resume before user commands/scripts, without scanning startup again."""
        if not self._paused:
            return
        # Freezing affects the entire interpreter. Only the application's own
        # entry point permits it; embedded callers and existing frozen heaps
        # retain their collector policy.
        if self._freeze_allowed and gc.get_freeze_count() == 0:
            gc.freeze()
            self._frozen = True
        gc.enable()
        self._paused = False
        if self._frozen:
            try:
                from Qt.QtCore import QTimer
                QTimer.singleShot(1000, self.unfreeze)
            except Exception:
                # An optional startup optimization must not prevent startup.
                self.unfreeze()

    def unfreeze(self):
        if self._frozen:
            gc.unfreeze()
            self._frozen = False

    def close(self):
        # Also restore on errors/early returns before resume(), or before the
        # timer runs. Do not override a user's later gc.disable() call.
        if self._paused:
            gc.enable()
            self._paused = False
        self.unfreeze()
