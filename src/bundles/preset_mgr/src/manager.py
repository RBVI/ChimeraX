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

from chimerax.core.state import StateManager
class PresetsManager(StateManager):
    """Manager for presets"""

    def __init__(self, session):
        self._alignments = {}
        self.session = session
        self._presets = {}
        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        self.triggers.add_trigger("presets changed")

    @property
    def presets_by_category(self):
        return {cat:[name for name in info.keys()] for cat,info in self._presets.items()}

    def preset_function(self, category, preset_name):
        return self._presets[category][preset_name]

    def remove_presets(self, category, preset_names):
        for name in preset_names:
            del self._presets[category][name]
        self.triggers.activate_trigger("presets changed", self)

    def add_presets(self, category, preset_info):
        """'preset_info' should be a dictionary of preset-name -> callback-function/command-string"""
        self._presets.setdefault(category, {}).update({
            name: lambda p=preset: self._call_preset(p) for name, preset in preset_info
        })
        self.triggers.activate_trigger("presets changed", self)

    def reset_state(self, session):
        pass

    @staticmethod
    def restore_snapshot(session, data):
        return session.presets

    def take_snapshot(self, session, flags):
        # Presets are "session enduring"
        return {}

    def _call_preset(self, preset):
        if callable(preset):
            preset()
            self.session.logger.info("Preset implemented in Python; no expansion to individual ChimeraX"
                " commands available.")
        else:
            from chimerax.core.commands import run
            run(self.session, preset, log=False)
            self.session.logger.info(
                "Preset expands to these ChimeraX commands: <i>%s</i>" % preset, is_html=True)
