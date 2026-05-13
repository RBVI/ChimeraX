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
            self.scenes = [
                scene for scene in self.scenes if scene.get_name() != scene_name
            ]
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

    @property
    def scene_names(self):
        return self.get_scene_names()

    @property
    def scene_relevant_models(self):
        return [m for m in self.session.models if m.SESSION_SAVE]

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
            "version": self.version,
            "scenes": [scene.take_snapshot(session, flags) for scene in self.scenes],
            "num_saved_scenes": self.num_saved_scenes,
        }

    @staticmethod
    def restore_snapshot(session, data):
        if data["version"] != SceneManager.version:
            raise ValueError(
                "scenes restore_snapshot: unknown version in data: %d" % data["version"]
            )
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
        for scene_snapshot in data["scenes"]:
            scene = Scene.restore_snapshot(self.session, scene_snapshot)
            self.scenes.append(scene)
        if "num_saved_scenes" in data:
            self.num_saved_scenes = data["num_saved_scenes"]

    def interpolate_scenes(
        self,
        scene1_name: str,
        scene2_name: str,
        fraction: float,
        *,
        seed: bool = False,
    ):
        """
        Interpolate between two scenes at the given fraction.

        Args:
            scene1_name (str): Name of the first scene
            scene2_name (str): Name of the second scene
            fraction (float): Interpolation fraction (0.0 = scene1, 1.0 = scene2)
            seed (bool): When True, restore scene1's full state before applying
                the interpolation. Needed when the caller can't guarantee that
                the scene graph already reflects scene1 — e.g. on the first
                frame of a transition or after a scrub jump — because some
                per-model interpolators only touch attributes present in both
                scenes and would otherwise leave stale values from wherever
                the user was previously parked.
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

        # Seed scene1's full state when the caller flags it. Without seeding,
        # attrs only present in scene1 (e.g. models hidden by scene2) keep
        # whatever value they had when the user last scrubbed elsewhere.
        # scene1.restore_scene() is called directly to avoid firing the
        # RESTORED trigger on every interpolation frame.
        if seed:
            scene1.restore_scene()

        # Get view data for interpolation
        v1 = scene1.named_view
        v2 = scene2.named_view
        current_view = self.session.view

        # Check if models actually moved between scenes or if only camera moved
        models_actually_moved = self._models_actually_moved(v1, v2)

        # Interpolate camera and model positions
        if models_actually_moved:
            # Models moved - use full interpolation including model positions
            from chimerax.std_commands.view import _interpolate_views

            # Calculate centers for model interpolation
            centers = {}
            models = self.session.scenes.scene_relevant_models
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
            from chimerax.std_commands.view import (
                _interpolate_camera,
                _interpolate_clip_planes,
            )
            _interpolate_camera(v1, v2, fraction, current_view.camera)
            _interpolate_clip_planes(v1, v2, fraction, current_view)

        # Interpolate ViewState appearance attributes (background, lighting,
        # material) that aren't covered by the camera/clip interpolation above.
        view_state = self.session.snapshot_methods(current_view)
        if view_state is not None and hasattr(view_state, 'interpolate_scene'):
            d1 = getattr(scene1, 'main_view_data', None)
            d2 = getattr(scene2, 'main_view_data', None)
            if d1 is not None and d2 is not None:
                view_state.interpolate_scene(current_view, self.session, d1, d2, fraction)

        # Interpolate model-specific scene data. A model that's hidden (or
        # absent) in one scene is treated as None on that side so the model's
        # own ``interpolate_scene`` can fade itself in or out — see
        # ``Structure.interpolate_scene`` for the atomic fade.
        scene1_visible = self._get_visible_models_in_scene(scene1)
        scene2_visible = self._get_visible_models_in_scene(scene2)
        for model in self.session.scenes.scene_relevant_models:
            info1 = scene1.scene_models.get(model)
            info2 = scene2.scene_models.get(model)
            if info1 is None and info2 is None:
                continue
            impl1, raw1 = info1 if info1 is not None else (False, None)
            impl2, raw2 = info2 if info2 is not None else (False, None)
            data1 = raw1 if (raw1 is not None and impl1 and model in scene1_visible) else None
            data2 = raw2 if (raw2 is not None and impl2 and model in scene2_visible) else None
            if data1 is None and data2 is None:
                continue
            model.interpolate_scene(data1, data2, fraction, switchover=(fraction >= 0.5))

    def _models_actually_moved(self, v1, v2):
        """
        Check if models actually moved between two views, or if only the camera moved.

        This is important to avoid unnecessary model interpolation when users are just
        rotating the camera around stationary models, which triggers expensive operations
        like ambient occlusion recalculation.
        """
        # print(f"DEBUG: _models_actually_moved called")

        # Get model positions from both views
        pos1 = v1.positions
        pos2 = v2.positions

        # print(f"DEBUG: pos1 has {len(pos1)} models, pos2 has {len(pos2)} models")

        # Don't consider model visibility changes as "movement"
        # Only check position changes of models that exist in BOTH scenes
        common_models = set(pos1.keys()) & set(pos2.keys())
        # print(f"DEBUG: {len(common_models)} models exist in both scenes")

        # If no common models, no movement to check
        if not common_models:
            # print(f"DEBUG: No common models - treating as camera-only movement")
            return False

        # Check if any model position actually changed
        # We need to be careful about floating point precision
        # Using a much more permissive tolerance for performance
        tolerance = (
            1e-1  # Very permissive - only catch actual intentional model movements
        )

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
            pass
            # print(f"DEBUG: Models actually moved between scenes - using full interpolation")
        else:
            pass
            # print(f"DEBUG: Only camera moved - using camera-only interpolation")

        return models_moved

    def _get_visible_models_in_scene(self, scene):
        """Get the set of models that are actually visible in a scene"""
        visible_models = set()

        # print(f"DEBUG: _get_visible_models_in_scene for scene '{scene.name if hasattr(scene, 'name') else 'unknown'}'")

        if not hasattr(scene, "named_view"):
            # print(f"DEBUG: Scene has no named_view")
            return visible_models

        if not hasattr(scene.named_view, "positions"):
            # print(f"DEBUG: NamedView has no positions")
            return visible_models

        # print(f"DEBUG: NamedView.positions has {len(scene.named_view.positions)} models")

        # Check what's in scene_models and if it contains display state
        # Commenting out verbose debug logging for now
        # if hasattr(scene, 'scene_models'):
        #     print(f"DEBUG: Scene.scene_models has {len(scene.scene_models)} models")
        #     for model, (has_restore, scene_data) in list(scene.scene_models.items())[:3]:  # Limit to first 3
        #         print(f"DEBUG: scene_models[{model}] = has_restore={has_restore}")
        #         print(f"DEBUG:   scene_data type: {type(scene_data)}")
        #         if isinstance(scene_data, dict):
        #             print(f"DEBUG:   scene_data keys: {list(scene_data.keys())}")
        #             for key, value in scene_data.items():
        #                 print(f"DEBUG:     {key}: {type(value)} = {value}")
        #                 if isinstance(value, dict):
        #                     print(f"DEBUG:       {key} sub-keys: {list(value.keys())}")
        #                     if 'display' in value:
        #                         print(f"DEBUG:         -> {key}.display = {value['display']}")
        #         else:
        #             print(f"DEBUG:   scene_data = {scene_data}")

        # Check display state from scene data for each current model
        current_models = self.session.scenes.scene_relevant_models
        for model in current_models:
            if hasattr(scene, "scene_models") and model in scene.scene_models:
                has_restore, scene_data = scene.scene_models[model]

                # Look for top-level display attribute in the scene data
                display_value = None
                if isinstance(scene_data, dict):
                    # Check for model state -> display (works for most models)
                    if "model state" in scene_data and isinstance(
                        scene_data["model state"], dict
                    ):
                        display_value = scene_data["model state"].get("display")
                    # For atomic structures, also check structure state -> model state -> display
                    elif "structure state" in scene_data and isinstance(
                        scene_data["structure state"], dict
                    ):
                        model_state = scene_data["structure state"].get(
                            "model state", {}
                        )
                        if isinstance(model_state, dict):
                            display_value = model_state.get("display")
                    # Volume scene data is a flat dict from state_from_map
                    # with no nested 'model state' — display sits at the top
                    # level alongside region, rendering_options, etc.
                    if display_value is None and "display" in scene_data:
                        display_value = scene_data["display"]

                if display_value is True:
                    visible_models.add(model)
                    # print(f"DEBUG: Model visible (display=True): {model}")
                elif display_value is False:
                    # print(f"DEBUG: Model hidden (display=False): {model}")
                    pass
                else:
                    # print(f"DEBUG: No display data found for: {model}")
                    pass
            else:
                # print(f"DEBUG: Model not in scene: {model}")
                pass

        # print(f"DEBUG: Returning {len(visible_models)} visible models")
        return visible_models
