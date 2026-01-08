"""
Scene Animation Manager for ChimeraX

This module provides a simplified animation system based on whole scenes,
similar to Chimera's approach. It manages scene transitions and interpolation
between complete scene states.
"""

from chimerax.core.state import StateManager
from chimerax.core.commands.motion import CallForNFrames
from Qt.QtCore import QObject, Signal as pyqtSignal, QTimer
from chimerax.core.commands.run import run
from chimerax.core.errors import UserError

import io
import math
import time
from typing import List, Tuple


class EasingFunctions:
    """Collection of easing functions for smooth transitions"""

    @staticmethod
    def linear(t):
        """Linear interpolation (no easing)"""
        return t

    @staticmethod
    def ease_in_sine(t):
        """Sine wave ease-in"""
        return 1.0 - math.cos((t * math.pi) / 2.0)

    @staticmethod
    def ease_out_sine(t):
        """Sine wave ease-out"""
        return math.sin((t * math.pi) / 2.0)

    @staticmethod
    def ease_in_out_sine(t):
        """Sine wave ease-in-out"""
        return -(math.cos(math.pi * t) - 1.0) / 2.0

    @staticmethod
    def ease_in_quad(t):
        """Quadratic ease-in"""
        return t * t

    @staticmethod
    def ease_out_quad(t):
        """Quadratic ease-out"""
        return 1.0 - (1.0 - t) * (1.0 - t)

    @staticmethod
    def ease_in_out_quad(t):
        """Quadratic ease-in-out"""
        return 2.0 * t * t if t < 0.5 else 1.0 - pow(-2.0 * t + 2.0, 2.0) / 2.0

    @staticmethod
    def ease_in_cubic(t):
        """Cubic ease-in"""
        return t * t * t

    @staticmethod
    def ease_out_cubic(t):
        """Cubic ease-out"""
        return 1.0 - pow(1.0 - t, 3.0)

    @staticmethod
    def ease_in_out_cubic(t):
        """Cubic ease-in-out"""
        return 4.0 * t * t * t if t < 0.5 else 1.0 - pow(-2.0 * t + 2.0, 3.0) / 2.0


# Transition types available to users
TRANSITION_TYPES = {
    "linear": EasingFunctions.linear,
    "ease_in_sine": EasingFunctions.ease_in_sine,
    "ease_out_sine": EasingFunctions.ease_out_sine,
    "ease_in_out_sine": EasingFunctions.ease_in_out_sine,
    "ease_in_quad": EasingFunctions.ease_in_quad,
    "ease_out_quad": EasingFunctions.ease_out_quad,
    "ease_in_out_quad": EasingFunctions.ease_in_out_quad,
    "ease_in_cubic": EasingFunctions.ease_in_cubic,
    "ease_out_cubic": EasingFunctions.ease_out_cubic,
    "ease_in_out_cubic": EasingFunctions.ease_in_out_cubic,
}

# Action types for rock/roll animations (these apply motion during transitions)
# Default configurations for action types
ACTION_DEFAULTS = {
    "rock": {"angle": 60, "axis": "y", "count": 1},  # Oscillate +/- angle degrees, count times
    "roll": {"angle": 360, "axis": "y", "count": 1},  # Rotate continuously
    "precess": {"axis": "y", "count": 1, "precession_tilt": 10},  # Wobble in cone around axis (no rotation)
}


class SceneAnimationSignals(QObject):
    """Signal emitter for SceneAnimation to avoid metaclass conflicts"""

    time_changed = pyqtSignal(float)  # Current playback time
    playback_started = pyqtSignal()
    playback_stopped = pyqtSignal()
    recording_started = pyqtSignal()
    recording_stopped = pyqtSignal()


class SceneAnimation(StateManager):
    """
    Manages scene-based animations with simple interpolation between scenes.
    This is a simplified alternative to the complex keyframe system.
    """

    version = 1
    DEFAULT_DURATION = 5.0
    DEFAULT_TRANSITION_TIME = 1.0

    def __init__(self, session, *, animation_data=None, fps=24):
        """Initialize the scene animation manager"""
        super().__init__()

        self.session = session
        self.logger = session.logger

        # Create signals object for Qt communication
        self.signals = SceneAnimationSignals()

        # Animation state
        self.duration = self.DEFAULT_DURATION
        self.scenes = []  # List of (time, scene_name, transition_data) tuples
        # transition_data = {'type': 'linear', 'fade_models': False}
        self.action_segments = []  # List of (start_time, end_time, action_name) tuples for rock/roll
        self.current_time = 0.0
        self.is_playing = False
        self.is_recording = False

        # Playback (use QTimer like keyframe system for consistency)
        self.fps = fps  # Match keyframe system FPS
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._advance_playback)
        self.start_time = 0.0
        self.reverse = False

        # Legacy support for recording
        self._call_for_n_frames = None
        self._record_data = None

        # Restore from snapshot if provided
        if animation_data:
            self.restore_from_data(animation_data)

    def add_scene_at_time(
        self,
        scene_name: str,
        time: float,
        transition_type: str = "linear",
        fade_models: bool = False,
        action: str = None,
    ):
        """Add a scene at a specific time with transition settings and optional action (rock/roll)"""
        if not self.session.scenes.get_scene(scene_name):
            #self.logger.warning(f"Scene '{scene_name}' does not exist")
            return False

        # Validate transition type
        if transition_type not in TRANSITION_TYPES:
                    #self.logger.warning(
            #f"Unknown transition type '{transition_type}', using 'linear'"
            #)
            transition_type = "linear"

        # Validate action type if provided
        if action and action not in ACTION_TYPES:
            #self.logger.warning(f"Unknown action type '{action}', ignoring")
            action = None

        # Remove any existing scene at this time
        self.scenes = [(t, s, td) for t, s, td in self.scenes if t != time]

        # Create transition data
        transition_data = {
            "type": transition_type,
            "fade_models": fade_models,
            "action": action  # Can be "rock", "roll", or None
        }

        # Add new scene
        self.scenes.append((time, scene_name, transition_data))
        self.scenes.sort(key=lambda x: x[0])  # Keep sorted by time

        #self.logger.info(
        #    f"Added scene '{scene_name}' at time {time:.2f}s with {transition_type} transition"
        #)
        return True

    def remove_scene(self, scene_name: str):
        """Remove all instances of a scene from the animation"""
        original_count = len(self.scenes)
        self.scenes = [(t, s, td) for t, s, td in self.scenes if s != scene_name]

        if len(self.scenes) < original_count:
            #self.logger.info(f"Removed scene '{scene_name}' from animation")
            return True
        else:
            #self.logger.warning(f"Scene '{scene_name}' not found in animation")
            return False

    def remove_scene_at_time(self, time: float):
        """Remove scene at specific time"""
        original_count = len(self.scenes)
        self.scenes = [(t, s, td) for t, s, td in self.scenes if t != time]

        if len(self.scenes) < original_count:
            #self.logger.info(f"Removed scene at time {time:.2f}s")
            return True
        else:
            #self.logger.warning(f"No scene found at time {time:.2f}s")
            return False

    def set_duration(self, duration: float):
        """Set the total duration of the animation"""
        if duration <= 0:
            #self.logger.warning("Duration must be positive")
            return False

        self.duration = duration

        # Remove any scenes beyond the new duration
        self.scenes = [(t, s, td) for t, s, td in self.scenes if t <= duration]

        #self.logger.info(f"Set animation duration to {duration:.2f}s")
        return True

    def get_effective_end_time(self):
        """Get the effective end time for recording (1 second after last scene)

        Returns the time 1 second after the last scene marker, or the full duration
        if there are no scenes.
        """
        if not self.scenes:
            return self.duration

        # Find the last scene time
        last_scene_time = max(t for t, _, _ in self.scenes)

        # Return 1 second after the last scene
        return last_scene_time + 1.0

    def preview_at_time(self, time: float):
        """Preview the animation at a specific time"""
        if time < 0 or time > self.duration:
            #self.logger.warning(f"Time {time:.2f}s is outside animation duration")
            return

        self.current_time = time

        # Find the appropriate scene or transition
        scene1, scene2, fraction = self._get_interpolation_at_time(time)

        # Apply any active action segments (rock/roll) at this time
        # This must happen BEFORE the early return check so actions work even when scenes don't change
        self._apply_action_segments(time)

        # Check if this is the same as what we're currently displaying to avoid redundant updates
        if hasattr(self, "_last_scene_state"):
            if self._last_scene_state == (scene1, scene2, fraction):
                return  # No change needed
            # Reset action angle tracking if we changed scenes
            if self._last_scene_state[:2] != (scene1, scene2):
                self._last_action_angle = 0.0

        if scene1 == scene2:
            # No interpolation needed, just restore the scene (faster)
            if scene1:  # Make sure scene exists
                self.session.scenes.restore_scene(scene1)

                # Check if we're at a scene timestamp that needs to prepare for model fading
                self._prepare_model_fading_at_scene_timestamp(scene1, time)
        else:
            # Interpolate between scenes (slower but necessary for transitions)
            if scene1 and scene2:
                # Debug: Show the easing effect
                # print(f"DEBUG: Interpolating {scene1} -> {scene2} at fraction {fraction:.3f}")

                # Check if model fading is enabled for the target scene
                scene2_data = self._get_scene_transition_data(scene2)
                fade_models = (
                    scene2_data.get("fade_models", False) if scene2_data else False
                )

                # print(f"DEBUG: scene2_data = {scene2_data}, fade_models = {fade_models}")

                # Pass fade_models flag to the scene manager
                self.session.scenes.interpolate_scenes(
                    scene1, scene2, fraction, fade_models=fade_models
                )

                # Apply action (rock/roll) if specified for the target scene
                action = scene2_data.get("action") if scene2_data else None
                if action:
                    self._apply_action(action, fraction)

        # Cache the current state to avoid redundant updates
        self._last_scene_state = (scene1, scene2, fraction)

        # Only log occasionally to avoid spam during playback
        if hasattr(self, "_last_log_time"):
            if time - self._last_log_time > 5.0:  # Log even less frequently
                #self.logger.info(f"Previewing animation at {time:.2f}s")
                self._last_log_time = time
        else:
            self._last_log_time = time

    def play(self, start_time: float = 0.0, reverse: bool = False):
        """Play the animation from start_time"""
        if self.is_playing:
            #self.logger.warning("Animation is already playing")
            return

        # Allow playback if we have either scenes OR action segments
        if not self.scenes and not self.action_segments:
            #self.logger.warning("No scenes or actions to animate")
            return

        if start_time < 0 or start_time > self.duration:
            #self.logger.warning(
            #f"Start time {start_time:.2f}s is outside animation duration"
            #)
            return

        self.is_playing = True
        self.current_time = start_time
        self.start_time = start_time
        self.reverse = reverse

        # Emit playback started signal
        self.signals.playback_started.emit()

        if self.is_recording:
            # When recording, don't use timer - advance only after frames are captured
                #self.logger.status(
                #    f"Recording animation at {self.fps} FPS (frame-synchronized)..."
                #)
            # Set up frame capture synchronization
            self._setup_recording_sync()
            # Start with the first frame
            self._advance_recording_frame()
        else:
            # Normal playback timing
            interval = int(1000 / self.fps)
            self.logger.status("Playing animation...")
            self.playback_timer.start(interval)

    def _advance_playback(self):
        """Advance animation by one frame (called by QTimer)"""
        if not self.is_playing:
            return

        # Calculate next time
        frame_duration = 1.0 / self.fps

        # Determine the effective end time
        # When recording, end 1 second after the last scene; otherwise use full duration
        end_time = self.get_effective_end_time() if self.is_recording else self.duration

        if self.reverse:
            next_time = self.current_time - frame_duration
            if next_time <= 0:
                next_time = 0
                self.stop_playing()
        else:
            next_time = self.current_time + frame_duration
            if next_time >= end_time:
                next_time = end_time
                self.stop_playing()

        # Update current time and preview
        self.current_time = next_time
        self.preview_at_time(next_time)

        # If recording, we need to wait for the frame to be drawn and captured
        # This will be handled by the frame_drawn trigger in the recording mode

        # Emit time change signal for UI updates
        self.signals.time_changed.emit(next_time)

    def _setup_recording_sync(self):
        """Set up frame capture synchronization for recording"""
        self._recording_frame_handler = None
        # Get the current frame count at start of recording
        self._last_frame_count = self.session.movie.getFrameCount()
        self._frame_wait_start_time = None

    def _advance_recording_frame(self):
        """Advance one frame during recording, synchronized with frame capture"""
        if not self.is_playing or not self.is_recording:
            return

        # Calculate next time
        frame_duration = 1.0 / self.fps

        # When recording, end 1 second after the last scene
        end_time = self.get_effective_end_time()

        if self.reverse:
            next_time = self.current_time - frame_duration
            if next_time <= 0:
                next_time = 0
                self.stop_playing()
                return
        else:
            next_time = self.current_time + frame_duration
            if next_time >= end_time:
                next_time = end_time
                self.stop_playing()
                return

        # Update current time and preview
        self.current_time = next_time
        self.preview_at_time(next_time)

        # Count expected frames
        self._expected_frame_count += 1

        # Emit time change signal for UI updates
        self.signals.time_changed.emit(next_time)

        # Wait for the frame to be captured, then advance to next frame
        self._wait_for_frame_capture()

    def _wait_for_frame_capture(self):
        """Wait for the current frame to be captured before advancing"""
        # Record when we started waiting for this frame
        import time

        self._frame_wait_start_time = time.time()
        # Set up a single-shot timer to check if frame was captured
        from Qt.QtCore import QTimer

        QTimer.singleShot(50, self._check_frame_captured)

    def _check_frame_captured(self):
        """Check if frame was captured and advance to next frame"""
        if not self.is_playing or not self.is_recording:
            return

        # Check if a new frame was captured
        current_frame_count = self.session.movie.getFrameCount()
        if current_frame_count > self._last_frame_count:
            # Frame was captured, advance to next
            self._last_frame_count = current_frame_count
            self._advance_recording_frame()
        else:
            # Check if we've been waiting too long (timeout after 500ms)
            import time

            if time.time() - self._frame_wait_start_time > 0.5:
                # Timeout - force a frame capture by manually triggering the movie system
                # print(f"DEBUG: Frame capture timeout, forcing frame capture (frame {current_frame_count + 1})")
                self.session.movie.capture_image()
                # Update our expected count and continue
                self._last_frame_count = self.session.movie.getFrameCount()
                self._advance_recording_frame()
            else:
                # Frame not captured yet, wait a bit more
                from Qt.QtCore import QTimer

                QTimer.singleShot(10, self._check_frame_captured)

    def stop_playing(self):
        """Stop playback"""
        # Stop QTimer
        self.playback_timer.stop()

        # Legacy support
        if self._call_for_n_frames:
            self._call_for_n_frames.done()
            self._call_for_n_frames = None

        self.is_playing = False

        # If we were recording, finish the recording
        if self.is_recording:
            self._finish_recording()

        # Emit playback stopped signal
        self.signals.playback_stopped.emit()
        #self.logger.status("Stopped animation")

    def set_fps(self, fps: int):
        """Update FPS and restart timer if playing"""
        self.fps = fps
        if self.is_playing:
            # Restart timer with new interval
            interval = int(1000 / self.fps)
            self.playback_timer.start(interval)

    def record(self, output_path: str, resolution=None, **kwargs):
        """Record the animation to a movie file

        Parameters:
        output_path: Path for the output video file
        resolution: Tuple of (width, height) for recording resolution, or string like '4k', '1080p'
                   If None, uses the setting from the animations preferences
        """
        if self.is_recording:
            #self.logger.warning("Already recording")
            return

        if not self.scenes:
            #self.logger.warning("No scenes to record")
            return

        try:
            # Start movie recording
            from chimerax.movie.moviecmd import movie_record, movie_encode

            # Use settings default if no resolution specified
            if resolution is None:
                from .settings import get_settings

                settings = get_settings(self.session)
                resolution = settings.recording_resolution

            # Parse resolution parameter
            size = self._parse_resolution(resolution)

            # Debug logging
            # print(f"DEBUG: Original resolution parameter: {resolution}")
            # print(f"DEBUG: Parsed size tuple: {size}")

            # Log recording info
            #if size:
            #    #self.logger.info(f"Recording at {size[0]}×{size[1]} resolution")
            #else:
            #    #self.logger.info("Recording at display resolution")

            # Set up recording parameters
            record_params = {
                "directory": None,  # Use temporary directory
                "pattern": None,  # Use default pattern
                "format": "png",
                "size": size,  # Set custom resolution
                **kwargs,
            }

            # Debug logging for movie record parameters
            # print(f"DEBUG: movie_record parameters: {record_params}")

            # Start recording
            movie_record(self.session, **record_params)
            self.is_recording = True
            self._record_data = {"output_path": output_path, "framerate": self.fps}
            self._expected_frame_count = 0  # Track expected frames during recording

            # Emit recording started signal
            self.signals.recording_started.emit()

            # Play the animation (this will generate frames)
            self.play()

        except Exception as e:
            #self.logger.error(f"Failed to start recording: {str(e)}")
            self.is_recording = False

    def _finish_recording(self):
        """Finish recording and encode movie"""
        if not self.is_recording:
            return

        try:
            from chimerax.movie.moviecmd import movie_encode

            run(self.session, "movie stop", log=False)

            # Get the number of frames that were actually captured
            actual_frame_count = self.session.movie.getFrameCount()
            # Calculate expected frames based on effective end time (1 second after last scene)
            effective_end_time = self.get_effective_end_time()
            expected_frames = getattr(
                self, "_expected_frame_count", effective_end_time * self.fps
            )

            # print(f"DEBUG: Animation duration: {self.duration}s at {self.fps} FPS")
            # print(f"DEBUG: Effective recording duration: {effective_end_time:.2f}s")
            # print(f"DEBUG: Expected frames during playback: {expected_frames}")
            # print(f"DEBUG: Actually captured by movie system: {actual_frame_count} frames")

            # Calculate what the actual video duration will be at the user's requested FPS
            actual_duration = actual_frame_count / self.fps
            # print(f"DEBUG: Video duration at {self.fps} FPS: {actual_duration:.2f}s")

            if abs(actual_frame_count - expected_frames) > 1:
                pass
                #self.logger.warning(
                #    f"Frame count mismatch: expected {expected_frames}, got {actual_frame_count}"
                #)

            # Always encode at the user's requested framerate
            movie_encode(
                self.session,
                output=[self._record_data["output_path"]],  # output should be a list
                framerate=self.fps,  # Use the user's requested FPS
            )

            #self.logger.info(
            #    f"Animation recorded to {self._record_data['output_path']}"
            #)

        except Exception as e:
            pass
            #self.logger.error(f"Failed to encode movie: {str(e)}")
        finally:
            self.is_recording = False
            self._record_data = None
            # Emit recording stopped signal
            self.signals.recording_stopped.emit()

    def _get_interpolation_at_time(self, time: float) -> Tuple[str, str, float]:
        """Get interpolation parameters for a specific time with easing"""
        if not self.scenes:
            return "", "", 0.0

        # Sort scenes by time (should already be sorted, but make sure)
        sorted_scenes = sorted(self.scenes, key=lambda x: x[0])

        # If only one scene, always show it
        if len(sorted_scenes) == 1:
            return sorted_scenes[0][1], sorted_scenes[0][1], 0.0

        # Before first scene - show first scene
        if time <= sorted_scenes[0][0]:
            return sorted_scenes[0][1], sorted_scenes[0][1], 0.0

        # After last scene - show last scene
        if time >= sorted_scenes[-1][0]:
            return sorted_scenes[-1][1], sorted_scenes[-1][1], 0.0

        # Find the two scenes we're between
        prev_scene = None
        next_scene = None

        for i in range(len(sorted_scenes) - 1):
            t1, scene1, transition1 = sorted_scenes[i]
            t2, scene2, transition2 = sorted_scenes[i + 1]

            if t1 <= time <= t2:
                prev_scene = (t1, scene1, transition1)
                next_scene = (t2, scene2, transition2)
                break

        if prev_scene is None or next_scene is None:
            # Fallback - shouldn't happen
            return sorted_scenes[0][1], sorted_scenes[0][1], 0.0

        # Extract scene data
        t1, scene1, transition1 = prev_scene
        t2, scene2, transition2 = next_scene

        # Debug: Show what transition data we found
        # print(f"DEBUG: Scene '{scene2}' has transition data: {transition2}")

        # Time between scenes
        time_between = t2 - t1

        if time_between == 0:
            # Scenes at same time, just show the later one
            return scene2, scene2, 0.0

        # How far through the transition are we?
        time_from_start = time - t1
        linear_fraction = time_from_start / time_between

        # Clamp linear fraction to [0, 1]
        linear_fraction = max(0.0, min(1.0, linear_fraction))

        # Apply easing function based on the target scene's transition type
        transition_type = transition2.get("type", "linear")
        easing_func = TRANSITION_TYPES.get(transition_type, TRANSITION_TYPES["linear"])
        eased_fraction = easing_func(linear_fraction)

        # Debug: Always show transition info
        # print(f"DEBUG: Transition type: {transition_type}, linear: {linear_fraction:.3f} -> eased: {eased_fraction:.3f}")

        return scene1, scene2, eased_fraction

    def _get_scene_transition_data(self, scene_name: str):
        """Get transition data for a scene by name"""
        for time, name, transition_data in self.scenes:
            if name == scene_name:
                return transition_data
        return None

    def _apply_action(self, action: str, fraction: float):
        """Apply rock/roll action during transition"""
        if action not in ACTION_TYPES:
            return

        action_config = ACTION_TYPES[action]
        action_type = action_config["type"]
        axis = action_config["axis"]
        angle = action_config["angle"]

        # Calculate rotation angle based on action type and fraction
        if action_type == "oscillate":  # Rock: oscillate back and forth
            # Use sine wave to oscillate: goes 0 -> max -> 0 -> -max -> 0
            rotation_angle = angle * math.sin(fraction * 2 * math.pi)
        elif action_type == "rotate":  # Roll: continuous rotation
            # Linear rotation from 0 to full angle
            rotation_angle = angle * fraction
        else:
            return

        # Calculate the incremental rotation since last frame
        if not hasattr(self, '_last_action_angle'):
            self._last_action_angle = 0.0

        delta_angle = rotation_angle - self._last_action_angle
        self._last_action_angle = rotation_angle

        # Apply incremental rotation to the view
        # Use ChimeraX's turn command to rotate the view
        if abs(delta_angle) > 0.01:  # Only apply if there's a meaningful change
            run(self.session, f"turn {axis} {delta_angle} center view")

    def _apply_action_segments(self, time: float):
        """Apply rock/roll actions from action segments at the current time"""
        # Check if we have any action segments
        if not self.action_segments:
            return

        # Find if we're in any action segment
        for segment_data in self.action_segments:
            start_time, end_time, action_name = segment_data[:3]
            config = segment_data[3] if len(segment_data) > 3 else ACTION_DEFAULTS.get(action_name, {})

            if start_time <= time <= end_time:
                # Calculate fraction within this segment
                segment_duration = end_time - start_time
                if segment_duration > 0:
                    fraction = (time - start_time) / segment_duration

                    # Get config parameters
                    angle = config.get("angle", 60)
                    axis = config.get("axis", "y")
                    count = config.get("count", 1)

                    # Get center of rotation from the current view
                    center = self.session.view.center_of_rotation

                    # Track state per segment to handle multiple segments
                    segment_key = (start_time, end_time, action_name)

                    if action_name == "precess":
                        # Precess: wobble camera in a cone around the axis without rotating
                        # Uses two perpendicular axes to create circular wobble pattern
                        precession_tilt = config.get("precession_tilt", 10)

                        # Determine the two perpendicular axes for wobbling
                        if axis == 'y':
                            wobble_axis1, wobble_axis2 = 'x', 'z'
                        elif axis == 'x':
                            wobble_axis1, wobble_axis2 = 'y', 'z'
                        else:  # axis == 'z'
                            wobble_axis1, wobble_axis2 = 'x', 'y'

                        # Calculate wobble angles using sin/cos to create circular motion
                        # count determines how many full wobble cycles
                        # Use (cos - 1) to ensure motion is periodic: starts and ends at zero
                        wobble_phase = fraction * count * 2 * math.pi
                        wobble_angle1 = precession_tilt * math.sin(wobble_phase)
                        wobble_angle2 = precession_tilt * (math.cos(wobble_phase) - 1.0)

                        # Track wobble angles separately for each axis
                        if not hasattr(self, '_wobble_angles'):
                            self._wobble_angles = {}

                        wobble_key1 = (segment_key, "wobble1")
                        wobble_key2 = (segment_key, "wobble2")

                        last_wobble1 = self._wobble_angles.get(wobble_key1, 0.0)
                        last_wobble2 = self._wobble_angles.get(wobble_key2, 0.0)

                        delta_wobble1 = wobble_angle1 - last_wobble1
                        delta_wobble2 = wobble_angle2 - last_wobble2

                        self._wobble_angles[wobble_key1] = wobble_angle1
                        self._wobble_angles[wobble_key2] = wobble_angle2

                        # Apply wobble rotations on both perpendicular axes
                        if abs(delta_wobble1) > 0.01:
                            run(self.session, f"turn {wobble_axis1} {delta_wobble1} center {center[0]},{center[1]},{center[2]}", log=False)
                        if abs(delta_wobble2) > 0.01:
                            run(self.session, f"turn {wobble_axis2} {delta_wobble2} center {center[0]},{center[1]},{center[2]}", log=False)

                    else:
                        # Rock and Roll: calculate rotation angle
                        if action_name == "rock":  # Oscillate
                            # count determines how many full oscillations (back and forth cycles)
                            rotation_angle = angle * math.sin(fraction * count * 2 * math.pi)
                        elif action_name == "roll":  # Rotate
                            # count determines how many full rotations
                            rotation_angle = angle * count * fraction
                        else:
                            continue

                        if not hasattr(self, '_segment_angles'):
                            self._segment_angles = {}

                        last_angle = self._segment_angles.get(segment_key, 0.0)
                        delta_angle = rotation_angle - last_angle
                        self._segment_angles[segment_key] = rotation_angle

                        # Apply incremental rotation around the center of rotation
                        if abs(delta_angle) > 0.01:
                            # Use silent log level to avoid spamming
                            run(self.session, f"turn {axis} {delta_angle} center {center[0]},{center[1]},{center[2]}", log=False)
                return  # Only apply one segment at a time

        # If we're not in any segment, reset tracking
        if hasattr(self, '_segment_angles'):
            self._segment_angles.clear()
        if hasattr(self, '_wobble_angles'):
            self._wobble_angles.clear()

    def _prepare_model_fading_at_scene_timestamp(
        self, current_scene_name: str, current_time: float
    ):
        """
        Prepare model fading when we're at an exact scene timestamp.
        This ensures that models appearing in the next scene with fade_models=True
        are made visible with zero opacity at the current scene's timestamp.
        """
        # Find if there's a next scene with model fading enabled
        sorted_scenes = sorted(self.scenes, key=lambda x: x[0])

        current_scene_index = None
        for i, (time, name, _) in enumerate(sorted_scenes):
            if (
                name == current_scene_name and abs(time - current_time) < 0.001
            ):  # Small tolerance for float comparison
                current_scene_index = i
                break

        if current_scene_index is None or current_scene_index >= len(sorted_scenes) - 1:
            # No next scene or this is the last scene
            return

        # Get the next scene
        next_time, next_scene_name, next_transition_data = sorted_scenes[
            current_scene_index + 1
        ]

        # Check if the next scene has model fading enabled
        fade_models = (
            next_transition_data.get("fade_models", False)
            if next_transition_data
            else False
        )

        if not fade_models:
            # Next scene doesn't have fading enabled
            return

        # print(f"DEBUG: Preparing model fading at scene '{current_scene_name}' timestamp {current_time:.2f}s for next scene '{next_scene_name}'")

        # Get scene objects
        current_scene = self.session.scenes.get_scene(current_scene_name)
        next_scene = self.session.scenes.get_scene(next_scene_name)

        if not current_scene or not next_scene:
            return

        # Find models that are visible in the next scene but not in the current scene
        current_visible_models = self._get_visible_models_in_scene(current_scene)
        next_visible_models = self._get_visible_models_in_scene(next_scene)

        appearing_models = next_visible_models - current_visible_models

        # print(f"DEBUG: Found {len(appearing_models)} models that will appear in next scene")

        # Make appearing models visible with zero opacity
        for model in appearing_models:
            if hasattr(model, "display"):
                # Make sure the model is visible but fully transparent
                model.display = True

                # For atomic models, we need to handle atoms.colors
                if hasattr(model, "atoms") and len(model.atoms) > 0:
                    atoms = model.atoms
                    # Get atom colors and set alpha to 0 (fully transparent)
                    c = atoms.colors
                    c[:, 3] = 0
                    atoms.colors = c
                    # print(f"DEBUG: Prepared atomic model for fade-in: set {len(atoms)} atom alphas to 0")

                # For non-atomic models, try the simple color approach
                elif hasattr(model, "color"):
                    try:
                        r, g, b, a = model.color
                        model.color = (r, g, b, 0)  # Fully transparent (0-255 range)
                        # print(f"DEBUG: Prepared non-atomic model for fade-in: set to visible with full transparency")
                    except:
                        pass
                    # print(f"DEBUG: Could not set color on model {model}")
                else:
                    pass
                # print(f"DEBUG: Model {model} has no atoms or color attribute")

    def _get_visible_models_in_scene(self, scene):
        """Get the set of models that are actually visible in a scene"""
        visible_models = set()

        if not hasattr(scene, "named_view") or not hasattr(
            scene.named_view, "positions"
        ):
            return visible_models

        # Models are visible in a scene if they have positions stored in the named_view
        # This follows the logic in scene.restore_scene() where models not in named_view.positions
        # get model.display = False (i.e., hidden)
        for model in scene.named_view.positions.keys():
            visible_models.add(model)

        return visible_models

    def get_scene_list(self) -> List[Tuple[float, str]]:
        """Get list of all scenes with their times (for compatibility)"""
        return [(t, s) for t, s, _ in self.scenes]

    def get_scene_list_with_transitions(self) -> List[Tuple[float, str, dict]]:
        """Get list of all scenes with their times and transition data"""
        return self.scenes.copy()

    def clear_all_scenes(self):
        """Remove all scenes from the animation"""
        self.scenes = []
        #self.logger.info("Cleared all scenes from animation")

    def take_snapshot(self, session, flags):
        """Save state for session snapshots"""
        return {
            "version": self.version,
            "duration": self.duration,
            "scenes": self.scenes,
            "current_time": self.current_time,
        }

    def restore_from_data(self, data):
        """Restore state from snapshot data"""
        if data.get("version", 0) != self.version:
            #self.logger.warning(f"Version mismatch in animation data")
            return

        self.duration = data.get("duration", self.DEFAULT_DURATION)
        self.scenes = data.get("scenes", [])
        self.current_time = data.get("current_time", 0.0)

        # Validate that all scenes still exist
        valid_scenes = []
        for time, scene_name, transition_data in self.scenes:
            if self.session.scenes.get_scene(scene_name):
                valid_scenes.append((time, scene_name, transition_data))
            else:
                pass
                #self.logger.warning(
                #    f"Scene '{scene_name}' no longer exists, removing from animation"
                #)

        self.scenes = valid_scenes

    @staticmethod
    def restore_snapshot(session, data):
        """Restore from session snapshot"""
        animation = SceneAnimation(session, animation_data=data)
        return animation

    def reset_state(self, session):
        """Reset to default state"""
        self.stop_playing()
        self.clear_all_scenes()
        self.duration = self.DEFAULT_DURATION
        self.current_time = 0.0
        self.is_recording = False

    def _parse_resolution(self, resolution):
        """Parse resolution parameter into (width, height) tuple"""
        if resolution is None:
            return None  # Use current display resolution

        if isinstance(resolution, tuple) and len(resolution) == 2:
            return resolution

        if isinstance(resolution, str):
            resolution = resolution.lower()
            if resolution in ["4k", "uhd", "2160p"]:
                return (3840, 2160)
            elif resolution in ["1080p", "fhd", "fullhd"]:
                return (1920, 1080)
            elif resolution in ["720p", "hd"]:
                return (1280, 720)
            elif resolution in ["480p", "sd"]:
                return (640, 480)
            else:
                    #self.logger.warning(
                #f"Unknown resolution '{resolution}', using display resolution"
                #)
                return None

            #self.logger.warning(
            #    f"Invalid resolution format '{resolution}', using display resolution"
            #)
        return None
