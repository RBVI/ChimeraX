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
from .triggers import activate_trigger, SAVED, DELETED, RENAMED, RESTORED


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
            scene_name = f"{self.num_saved_scenes + 1}"
        if self.scene_exists(scene_name):
            self.get_scene(scene_name).init_from_session()
        else:
            self.scenes.append(Scene(self.session, scene_name))
            self.num_saved_scenes += 1
        activate_trigger(SAVED, scene_name)

    def restore_scene(self, scene_name):
        """
        Restore a scene by name.
        """
        if self.scene_exists(scene_name):
            self.get_scene(scene_name).restore_scene()
            activate_trigger(RESTORED, scene_name)
        return

    def rename_scene(self, scene_name, new_scene_name):
        """
        Rename a scene.
        """
        if self.scene_exists(new_scene_name):
            self.session.logger.warning(f"Scene {new_scene_name} already exists.")
            return
        if self.scene_exists(scene_name):
            self.get_scene(scene_name).rename_scene(new_scene_name)
            activate_trigger(RENAMED, (scene_name, new_scene_name))
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

    def interpolate_scenes(self, scene1_name: str, scene2_name: str, fraction: float):
        """
        Interpolate between two scenes at the given fraction.

        Args:
            scene1_name (str): Name of the first scene
            scene2_name (str): Name of the second scene
            fraction (float): Interpolation fraction (0.0 = scene1, 1.0 = scene2)
        """
        scene1 = self.get_scene(scene1_name)
        scene2 = self.get_scene(scene2_name)

        if not scene1:
            self.session.logger.warning(f"Scene '{scene1_name}' not found")
            return
        if not scene2:
            self.session.logger.warning(f"Scene '{scene2_name}' not found")
            return

        # Clamp fraction to valid range
        fraction = max(0.0, min(1.0, fraction))

        # If fraction is 0, just restore scene1
        if fraction == 0.0:
            self.restore_scene(scene1_name)
            return
        # If fraction is 1, just restore scene2
        elif fraction == 1.0:
            self.restore_scene(scene2_name)
            return

        # Import interpolation functions from view module
        from chimerax.std_commands.view import _interpolate_views

        # Create mock view objects for interpolation
        # We'll use the NamedView data to interpolate camera and model positions
        v1 = scene1.named_view
        v2 = scene2.named_view

        # Get current view to apply interpolation to
        current_view = self.session.view

        # Calculate centers for model interpolation (needed for _interpolate_views)
        centers = {}
        models = self.session.models.list()
        for model in models:
            if model in v1.positions and model in v2.positions:
                # Use model center for interpolation, handle None bounds
                bounds = model.bounds()
                if bounds is not None:
                    centers[model] = bounds.center()
                else:
                    # Use origin if model has no bounds
                    from chimerax.geometry import Point
                    centers[model] = Point(0, 0, 0)

        # Perform the interpolation
        _interpolate_views(v1, v2, fraction, current_view, centers)

        # Interpolate model-specific scene data if models support it
        current_models = self.session.models.list()
        for model in current_models:
            # Check if both scenes have data for this model
            if (model in scene1.scene_models and model in scene2.scene_models):
                scene1_restore_implemented, scene1_data = scene1.scene_models[model]
                scene2_restore_implemented, scene2_data = scene2.scene_models[model]

                # Only interpolate if both models have proper scene restore support
                if scene1_restore_implemented and scene2_restore_implemented:
                    # For now, just use a simple approach: apply scene1 data if fraction < 0.5, else scene2
                    # More sophisticated model property interpolation could be implemented later
                    if fraction < 0.5:
                        if hasattr(model, 'restore_scene'):
                            model.restore_scene(scene1_data)
                        else:
                            from chimerax.core.models import Model
                            Model.restore_scene(model, scene1_data)
                    else:
                        if hasattr(model, 'restore_scene'):
                            model.restore_scene(scene2_data)
                        else:
                            from chimerax.core.models import Model
                            Model.restore_scene(model, scene2_data)

