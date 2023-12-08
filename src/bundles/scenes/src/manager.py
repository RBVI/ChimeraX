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

from chimerax.core.state import StateManager
class SceneManager(StateManager):
    """Manager for scenes"""

    ADDED, DELETED = trigger_names = ("added", "deleted")

    def __init__(self, session, bundle_info):
        self.scenes = {} # name -> Scene
        self.session = session
        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        for trig_name in self.trigger_names:
            self.triggers.add_trigger(trig_name)
        from chimerax.core.models import REMOVE_MODELS
        session.triggers.add_handler(REMOVE_MODELS, self._remove_models_cb)

    def delete_scene(self, scene_name):
        del self.scenes[scene_name]
        self.triggers.activate_trigger(self.DELETED, scene_name)

    def clear(self):
        for scene_name in list(self.scene.keys()):
            self.delete_scene(scene_name)

    def save_scene(self, scene_name):
        """Save scene named 'scene_name'"""
        if scene_name in self.scenes:
            self.delete_scene(scene_name)
        from .scene import Scene
        self.scenes[scene_name] = Scene(self.session)

    def _remove_models_cb(self, trig_name, models):
        for scene in self.scenes.values:
            scene.models_removed(models)

    # session methods
    def reset_state(self, session):
        self.clear()

    @staticmethod
    def restore_snapshot(session, data):
        mgr = session.scene
        mgr._ses_restore(data)
        return mgr

    def take_snapshot(self, session, flags):
        # viewer_info is "session independent"
        return {
            'version': 1,
            'scenes': self.scenes,
        }

    def _ses_restore(self, data):
        self.clear()
        self.scenes = data['scenes']
        for scene_name in self.scenes.keys():
            self.triggers.activate(self.ADDED, scene_name)
