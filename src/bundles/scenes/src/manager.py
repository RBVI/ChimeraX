# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2025 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# You can also
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

from typing import Optional
from chimerax.core.state import StateManager
from .scene import Scene
from chimerax.core.models import REMOVE_MODELS
from .triggers import activate_trigger, SAVED, DELETED, EDITED


class SceneManager(StateManager):
    """
    Manager for scenes in ChimeraX.

    This class manages the creation, deletion, editing, saving, and restoring of scenes. It also handles the
    removal of models from scenes and provides methods to reset the state and take or restore snapshots.

    Attributes:
        version (int): The version of the SceneManager.
        scenes (dict): A dictionary mapping scene names to Scene objects.
        session: The current session.
    """

    version = 0

    def __init__(self, session):
        """
        Initialize the SceneManager.

        Args:
            session: The current session.
        """
        self.scenes: [Scene] = []
        self.session = session
        self.num_saved_scenes = 0
        session.triggers.add_handler(REMOVE_MODELS, self._remove_models_cb)

    def scene_exists(self, scene_name: str) -> bool:
        """
        Check if a scene exists by name and return True if it does, False otherwise.
        """
        return scene_name in [scene.get_name() for scene in self.scenes]

    def delete_scene(self, scene_name):
        """
        Delete scene by name.
        """
        if self.scene_exists(scene_name):
            self.scenes = [scene for scene in self.scenes if scene.get_name() != scene_name]
            activate_trigger(DELETED, scene_name)
        else:
            self.session.logger.warning(f"Scene {scene_name} does not exist.")

    def edit_scene(self, scene_name):
        """
        Edit a scene by name. This method will re-initialize the scene from the current session state.
        """
        if self.scene_exists(scene_name):
            self.get_scene(scene_name).init_from_session()
            activate_trigger(EDITED, scene_name)
        else:
            self.session.logger.warning(f"Scene {scene_name} does not exist.")

    def clear(self):
        """
        Delete all scenes.
        """
        for scene in self.scenes:
            self.delete_scene(scene.get_name())

    def save_scene(self, scene_name: Optional[str] = None) -> None:
        """
        Save the current state as a scene.
        """
        if not scene_name:
            scene_name = f"Scene {self.num_saved_scenes + 1}"
        if self.scene_exists(scene_name):
            self.session.logger.warning(f"Scene {scene_name} already exists.")
            return
        self.scenes.append(Scene(self.session, scene_name))
        self.num_saved_scenes += 1
        activate_trigger(SAVED, scene_name)

    def restore_scene(self, scene_name):
        """
        Restore a scene by name.
        """
        if self.scene_exists(scene_name):
            self.get_scene(scene_name).restore_scene()
        return

    # session methods
    def reset_state(self, session):
        """
        Reset the state of the SceneManager by removing all the scenes.
        """
        self.clear()

    def get_scenes(self):
        return self.scenes

    def get_scene(self, scene_name: str) -> Scene | None:
        """
        Get a scene by name. If the scene does not exist, return None.
        """
        for scene in self.scenes:
            if scene.get_name() == scene_name:
                return scene
        return None

    def get_scene_names(self):
        """
        Returns:
            list[str]: Array of scene names.
        """
        return [scene.get_name() for scene in self.scenes]

    def _remove_models_cb(self, trig_name, models):
        """
        Callback for removing models from scenes.

        Args:
            trig_name (str): The name of the trigger.
            models: The models to remove.
        """
        for scene in self.scenes:
            scene.models_removed(models)

    def take_snapshot(self, session, flags):
        # viewer_info is "session independent"
        return {
            'version': self.version,
            'scenes': [scene.take_snapshot(session, flags) for scene in self.scenes],
            'num_saved_scenes': self.num_saved_scenes
        }

    @staticmethod
    def restore_snapshot(session, data):
        if data['version'] != SceneManager.version:
            raise ValueError("scenes restore_snapshot: unknown version in data: %d" % data['version'])
        mgr = session.scenes
        mgr._restore_snapshot(data)
        return mgr

    def _restore_snapshot(self, data):
        """
        Restore the SceneManager scenes attribute from session data.

        Args:
            data (dict): The session data.
        """
        self.clear()
        for scene_snapshot in data['scenes']:
            scene = Scene.restore_snapshot(self.session, scene_snapshot)
            self.scenes.append(scene)
        if 'num_saved_scenes' in data:
            self.num_saved_scenes = data['num_saved_scenes']

