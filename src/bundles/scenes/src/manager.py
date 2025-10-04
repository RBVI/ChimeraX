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

    def interpolate_scenes(self, scene1_name: str, scene2_name: str, fraction: float, fade_models: bool = False):
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

        # Get view data for interpolation
        v1 = scene1.named_view
        v2 = scene2.named_view
        current_view = self.session.view

        # Check if models actually moved between scenes or if only camera moved
        models_actually_moved = self._models_actually_moved(v1, v2)

        print(f"DEBUG: Interpolating between '{scene1_name}' and '{scene2_name}' at fraction {fraction}")
        print(f"DEBUG: Number of models in scene: {len(self.session.models.list())}")

        if models_actually_moved:
            # Models moved - use full interpolation including model positions
            from chimerax.std_commands.view import _interpolate_views

            # Calculate centers for model interpolation
            centers = {}
            models = self.session.models.list()
            for model in models:
                if model in v1.positions and model in v2.positions:
                    bounds = model.bounds()
                    if bounds is not None:
                        centers[model] = bounds.center()
                    else:
                        import numpy as np
                        centers[model] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

            # Perform full interpolation (camera + models)
            _interpolate_views(v1, v2, fraction, current_view, centers)
        else:
            # Only camera moved - interpolate only camera and clip planes
            # This avoids moving models and triggering expensive ambient occlusion updates
            from chimerax.std_commands.view import _interpolate_camera, _interpolate_clip_planes
            _interpolate_camera(v1, v2, fraction, current_view.camera)
            _interpolate_clip_planes(v1, v2, fraction, current_view)

        # Only interpolate model-specific scene data if models actually moved
        if models_actually_moved:
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

        # Handle model fading if enabled
        if fade_models:
            print(f"DEBUG: Applying model fade effects")
            self._apply_model_fade(scene1, scene2, fraction)

    def _apply_model_fade(self, scene1, scene2, fraction):
        """Apply fade in/out effects for models appearing/disappearing between scenes"""
        # Get models present in each scene
        scene1_models = set(scene1.scene_models.keys()) if hasattr(scene1, 'scene_models') else set()
        scene2_models = set(scene2.scene_models.keys()) if hasattr(scene2, 'scene_models') else set()

        # Models that appear in scene2 but not scene1 (need to fade in)
        appearing_models = scene2_models - scene1_models
        # Models that disappear (in scene1 but not scene2) (need to fade out)
        disappearing_models = scene1_models - scene2_models

        print(f"DEBUG: Scene1 models: {len(scene1_models)}, Scene2 models: {len(scene2_models)}")
        print(f"DEBUG: Appearing: {len(appearing_models)}, Disappearing: {len(disappearing_models)}")

        for model in appearing_models:
            # Model should fade in: opacity goes from 0 to original opacity
            if hasattr(model, 'transparency'):
                # Get target transparency from scene2 (default to opaque if not stored)
                target_transparency = 0  # Assume opaque as default
                if model in scene2.scene_models:
                    # Try to extract transparency from scene data if available
                    _, scene2_data = scene2.scene_models[model]
                    # For now, use default opaque. Scene data parsing would be more complex.

                # Fade in: start fully transparent, end at target transparency
                current_transparency = int(255 * (1.0 - fraction) + target_transparency * fraction)
                model.transparency = current_transparency
                print(f"DEBUG: Fading in model: transparency {current_transparency}")

        for model in disappearing_models:
            # Model should fade out: opacity goes from original to 0
            if hasattr(model, 'transparency'):
                # Get original transparency from scene1
                original_transparency = 0  # Assume opaque as default
                if model in scene1.scene_models:
                    # Try to extract transparency from scene data if available
                    _, scene1_data = scene1.scene_models[model]
                    # For now, use default opaque

                # Fade out: start at original transparency, end fully transparent
                current_transparency = int(original_transparency * (1.0 - fraction) + 255 * fraction)
                model.transparency = current_transparency
                print(f"DEBUG: Fading out model: transparency {current_transparency}")

    def _models_actually_moved(self, v1, v2):
        """
        Check if models actually moved between two views, or if only the camera moved.

        This is important to avoid unnecessary model interpolation when users are just
        rotating the camera around stationary models, which triggers expensive operations
        like ambient occlusion recalculation.
        """
        print(f"DEBUG: _models_actually_moved called")

        # Get model positions from both views
        pos1 = v1.positions
        pos2 = v2.positions

        print(f"DEBUG: pos1 has {len(pos1)} models, pos2 has {len(pos2)} models")

        # Don't consider model visibility changes as "movement"
        # Only check position changes of models that exist in BOTH scenes
        common_models = set(pos1.keys()) & set(pos2.keys())
        print(f"DEBUG: {len(common_models)} models exist in both scenes")

        # If no common models, no movement to check
        if not common_models:
            print(f"DEBUG: No common models - treating as camera-only movement")
            return False

        # Check if any model position actually changed
        # We need to be careful about floating point precision
        # Using a much more permissive tolerance for performance
        tolerance = 1e-1  # Very permissive - only catch actual intentional model movements

        models_moved = False
        for model in common_models:  # Only check models in both scenes

            positions1 = pos1[model]
            positions2 = pos2[model]

            # If different number of positions, definitely moved
            if len(positions1) != len(positions2):
                models_moved = True
                break

            # Compare each position with tolerance
            for i, (p1, p2) in enumerate(zip(positions1, positions2)):
                # Get transformation matrices for comparison
                m1 = p1.matrix
                m2 = p2.matrix

                # Check if matrices are significantly different
                import numpy as np
                diff = np.abs(m1 - m2)
                max_diff = np.max(diff)
                if max_diff > tolerance:
                    models_moved = True
                    break

            if models_moved:
                break

        # Debug logging to understand performance issues
        if models_moved:
            print(f"DEBUG: Models actually moved between scenes - using full interpolation")
        else:
            print(f"DEBUG: Only camera moved - using camera-only interpolation")

        return models_moved

