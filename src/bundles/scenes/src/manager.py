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
        fade_models: bool = False,
    ):
        print(f"DEBUG: interpolate_scenes called: {scene1_name} -> {scene2_name}, fraction={fraction:.3f}")
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

        # print(
        # f"DEBUG: Interpolating between '{scene1_name}' and '{scene2_name}' at fraction {fraction}"
        # )
        # print(f"DEBUG: Number of models in scene: {len(self.session.models.list())}")

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
            from chimerax.std_commands.view import (
                _interpolate_camera,
                _interpolate_clip_planes,
            )

            _interpolate_camera(v1, v2, fraction, current_view.camera)
            _interpolate_clip_planes(v1, v2, fraction, current_view)

        # Always check for volume interpolation regardless of whether models moved
        current_models = self.session.models.list()
        print(f"DEBUG: Checking {len(current_models)} models for volume interpolation")
        for model in current_models:
            # Check if both scenes have data for this model
            if model in scene1.scene_models and model in scene2.scene_models:
                scene1_restore_implemented, scene1_data = scene1.scene_models[model]
                scene2_restore_implemented, scene2_data = scene2.scene_models[model]

                # Only interpolate if both models have proper scene restore support
                if scene1_restore_implemented and scene2_restore_implemented:
                    # Check if this is a volume model and handle smooth interpolation
                    print(f"DEBUG: Checking model {model} ({type(model)})")
                    print(f"DEBUG: Model module: {model.__class__.__module__}")
                    print(f"DEBUG: Has new_region: {hasattr(model, 'new_region')}")
                    print(f"DEBUG: Has region: {hasattr(model, 'region')}")
                    print(f"DEBUG: Has data: {hasattr(model, 'data')}")
                    print(f"DEBUG: Is volume model: {self._is_volume_model(model)}")

                    if self._is_volume_model(model):
                        print(f"DEBUG: Found volume model {model}, calling interpolation")
                        self._interpolate_volume_model(model, scene1_data, scene2_data, fraction)
                    else:
                        print(f"DEBUG: Not identified as volume model: {model}")

        # Only interpolate non-volume model-specific scene data if models actually moved
        if models_actually_moved:
            print(f"DEBUG: Models actually moved, starting non-volume model interpolation")
            for model in current_models:
                # Check if both scenes have data for this model
                if model in scene1.scene_models and model in scene2.scene_models:
                    scene1_restore_implemented, scene1_data = scene1.scene_models[model]
                    scene2_restore_implemented, scene2_data = scene2.scene_models[model]

                    # Only interpolate if both models have proper scene restore support
                    if scene1_restore_implemented and scene2_restore_implemented:
                        # Skip volume models (already handled above)
                        if not self._is_volume_model(model):
                            print(f"DEBUG: Non-volume model using threshold approach: {model}")
                            # For non-volume models, use simple threshold approach
                            if fraction < 0.5:
                                if hasattr(model, "restore_scene"):
                                    model.restore_scene(scene1_data)
                                else:
                                    from chimerax.core.models import Model
                                    Model.restore_scene(model, scene1_data)
                            else:
                                if hasattr(model, "restore_scene"):
                                    model.restore_scene(scene2_data)
                                else:
                                    from chimerax.core.models import Model
                                    Model.restore_scene(model, scene2_data)

        # Handle model fading if enabled
        if fade_models:
            # print(f"DEBUG: Applying model fade effects")
            self._apply_model_fade(scene1, scene2, fraction)

    def _apply_model_fade(self, scene1, scene2, fraction):
        """Apply fade in/out effects for models appearing/disappearing between scenes"""
        # Get models that are visible in each scene (not just present in scene data)
        scene1_visible_models = self._get_visible_models_in_scene(scene1)
        scene2_visible_models = self._get_visible_models_in_scene(scene2)

        # Models that become visible in scene2 but were not visible in scene1 (need to fade in)
        appearing_models = scene2_visible_models - scene1_visible_models
        # Models that were visible in scene1 but are not visible in scene2 (need to fade out)
        disappearing_models = scene1_visible_models - scene2_visible_models

        # print(f"DEBUG: Scene1 visible models: {len(scene1_visible_models)}, Scene2 visible models: {len(scene2_visible_models)}")
        # print(f"DEBUG: Appearing: {len(appearing_models)}, Disappearing: {len(disappearing_models)}")

        # IMPORTANT: Make models visible during transition if they're fading in either direction
        # This ensures the fade effect is actually visible during animation
        models_to_fade = appearing_models | disappearing_models
        for model in models_to_fade:
            if hasattr(model, "display"):
                model.display = True
                # print(f"DEBUG: Made model visible for fading: {model}")

        # Debug: Show current session models and their display state
        # current_session_models = list(self.session.models.list())
        # print(f"DEBUG: Current session has {len(current_session_models)} models:")
        # for model in current_session_models[:5]:  # Limit to first 5
        # print(f"DEBUG: Session model: {model}, display={getattr(model, 'display', 'N/A')}")

        # Debug: Let's see what models we actually have
        # print(f"DEBUG: Scene1 visible model names: {[str(m) for m in scene1_visible_models]}")
        # print(f"DEBUG: Scene2 visible model names: {[str(m) for m in scene2_visible_models]}")

        # Debug: Let's manually check the difference
        # print(f"DEBUG: Models only in Scene1: {[str(m) for m in scene1_visible_models - scene2_visible_models]}")
        # print(f"DEBUG: Models only in Scene2: {[str(m) for m in scene2_visible_models - scene1_visible_models]}")

        if appearing_models:
            # print(f"DEBUG: Appearing model names: {[str(m) for m in appearing_models]}")
            pass
        if disappearing_models:
            # print(f"DEBUG: Disappearing model names: {[str(m) for m in disappearing_models]}")
            pass

        # If no models to fade, explain why
        if len(appearing_models) == 0 and len(disappearing_models) == 0:
            # print(f"DEBUG: No models to fade - both scenes have identical visible models.")
            # print(f"DEBUG: For model fading to work, scenes must have different model visibility.")
            # print(f"DEBUG: Hide some models before saving one scene, then show them before saving the other.")
            pass

        for model in appearing_models:
            # Model should fade in: opacity goes from 0 to original opacity
            # print(f"DEBUG: Checking appearing model {model}")

            # For atomic models, we need to handle atoms.colors
            if hasattr(model, "atoms") and len(model.atoms) > 0:
                atoms = model.atoms
                # print(f"DEBUG: Atomic model with {len(atoms)} atoms")

                # Get target transparency from scene2 (default to opaque if not stored)
                target_alpha = 255  # Assume opaque as default (0-255 range)
                if model in scene2.scene_models:
                    # Try to extract transparency from scene data if available
                    _, scene2_data = scene2.scene_models[model]
                    # For now, use default opaque. Scene data parsing would be more complex.

                # Fade in: start fully transparent, end at target transparency
                current_alpha = int(0 * (1.0 - fraction) + target_alpha * fraction)

                # Get atom colors and modify alpha channel
                c = atoms.colors
                c[:, 3] = current_alpha
                atoms.colors = c
                # print(f"DEBUG: Fading in atomic model: set {len(atoms)} atom alphas to {current_alpha}")

            # For non-atomic models, try the simple color approach
            elif hasattr(model, "color"):
                # Get target transparency from scene2 (default to opaque if not stored)
                target_alpha = 255  # Assume opaque as default (0-255 range)
                current_alpha = int(0 * (1.0 - fraction) + target_alpha * fraction)

                # For non-atomic models, color might be a simple tuple
                try:
                    r, g, b, a = model.color
                    model.color = (r, g, b, current_alpha)
                    # print(f"DEBUG: Fading in non-atomic model: set color alpha to {current_alpha}")
                except:
                    # print(f"DEBUG: Could not set color on model {model}")
                    pass
            else:
                # print(f"DEBUG: Model {model} has no atoms or color attribute")
                pass

        for model in disappearing_models:
            # Model should fade out: opacity goes from original to 0
            # print(f"DEBUG: Checking disappearing model {model}")

            # For atomic models, we need to handle atoms.colors
            if hasattr(model, "atoms") and len(model.atoms) > 0:
                atoms = model.atoms
                # print(f"DEBUG: Atomic model with {len(atoms)} atoms")

                # Get original transparency from scene1 (default to opaque)
                original_alpha = 255  # Assume opaque as default (0-255 range)
                if model in scene1.scene_models:
                    # Try to extract transparency from scene data if available
                    _, scene1_data = scene1.scene_models[model]
                    # For now, use default opaque

                # Fade out: start at original transparency, end fully transparent
                current_alpha = int(original_alpha * (1.0 - fraction) + 0 * fraction)

                # Get atom colors and modify alpha channel
                c = atoms.colors
                c[:, 3] = current_alpha
                atoms.colors = c
                # print(f"DEBUG: Fading out atomic model: set {len(atoms)} atom alphas to {current_alpha}")

            # For non-atomic models, try the simple color approach
            elif hasattr(model, "color"):
                # Get original transparency from scene1 (default to opaque)
                original_alpha = 255  # Assume opaque as default (0-255 range)
                current_alpha = int(original_alpha * (1.0 - fraction) + 0 * fraction)

                # For non-atomic models, color might be a simple tuple
                try:
                    r, g, b, a = model.color
                    model.color = (r, g, b, current_alpha)
                    # print(f"DEBUG: Fading out non-atomic model: set color alpha to {current_alpha}")
                except:
                    pass
                # print(f"DEBUG: Could not set color on model {model}")
            else:
                pass
            # print(f"DEBUG: Model {model} has no atoms or color attribute")

    def _apply_model_alpha(self, model, start_alpha, end_alpha, fraction):
        """Apply alpha interpolation to a model (works for both fade in/out)"""
        current_alpha = int(start_alpha * (1.0 - fraction) + end_alpha * fraction)

        if hasattr(model, "atoms") and len(model.atoms) > 0:
            c = model.atoms.colors
            c[:, 3] = current_alpha
            model.atoms.colors = c
        elif hasattr(model, "color"):
            try:
                r, g, b, a = model.color
                model.color = (r, g, b, current_alpha)
            except:
                pass

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
        current_models = self.session.models.list()
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

    def _is_volume_model(self, model):
        """Check if a model is a Volume model from the map bundle"""
        # Check if this is a Volume model without importing the map bundle
        return (hasattr(model, 'new_region') and
                hasattr(model, 'region') and
                hasattr(model, 'data') and
                model.__class__.__module__.startswith('chimerax.map'))

    def _interpolate_volume_model(self, model, scene1_data, scene2_data, fraction):
        """Interpolate volume model between two scene states"""
        # Call volume's restore_scene method directly since it handles volume-specific data
        model.restore_scene(scene1_data)

        # Get volume states from both scenes - they should be directly in scene data
        volume_state1 = scene1_data.get('volume state', scene1_data)
        volume_state2 = scene2_data.get('volume state', scene2_data)

        # Debug: Print what we're working with
        print(f"DEBUG: Volume interpolation called for {model}, fraction={fraction:.3f}")
        print(f"DEBUG: scene1_data keys: {scene1_data.keys()}")
        print(f"DEBUG: scene2_data keys: {scene2_data.keys()}")

        # Interpolate volume region (the main feature)
        region1 = volume_state1.get('region')
        region2 = volume_state2.get('region')

        if region1 and region2:
            print(f"DEBUG: region1={region1}")
            print(f"DEBUG: region2={region2}")

            # Early exit if regions are the same
            if region1 == region2:
                print(f"DEBUG: Regions are identical, skipping interpolation")
                return

            # Extract region parameters
            ijk_min1, ijk_max1, ijk_step1 = region1
            ijk_min2, ijk_max2, ijk_step2 = region2

            # Interpolate region bounds
            ijk_min_interp = [
                int(round(min1 + fraction * (min2 - min1)))
                for min1, min2 in zip(ijk_min1, ijk_min2)
            ]
            ijk_max_interp = [
                int(round(max1 + fraction * (max2 - max1)))
                for max1, max2 in zip(ijk_max1, ijk_max2)
            ]

            # For step size, use threshold behavior since interpolating steps can be problematic
            ijk_step_interp = ijk_step2 if fraction >= 0.5 else ijk_step1

            print(f"DEBUG: Interpolated region: min={ijk_min_interp}, max={ijk_max_interp}, step={ijk_step_interp}")

            # Apply the interpolated region
            print(f"DEBUG: Applying interpolated region to volume")
            model.new_region(ijk_min_interp, ijk_max_interp, ijk_step_interp,
                           adjust_step=False, adjust_voxel_limit=False)

            # Make sure volume updates its rendering
            model._drawings_need_update()
            print(f"DEBUG: Volume region interpolation complete")
            return
        else:
            print(f"DEBUG: No region data found for interpolation, using threshold behavior")
            # Fall back to threshold behavior for non-region properties
            if fraction >= 0.5:
                from chimerax.map.session import set_map_state
                set_map_state(volume_state2, model, notify=True)
            # else keep scene1 which was already applied by restore_scene
