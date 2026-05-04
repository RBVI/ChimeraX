"""
Scene Animation Manager for ChimeraX

This module provides a simplified animation system based on whole scenes,
similar to Chimera's approach. It manages scene transitions and interpolation
between complete scene states.
"""

from chimerax.core.state import StateManager
from chimerax.core.commands.motion import CallForNFrames
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
    "precess": {"angle": 30, "axis": "y", "count": 1, "wobble_aspect": 0.3},  # Figure-8 wobble matching ChimeraX wobble command
}


_LEGACY_TRANSITION_KEYS = ("fade_models",)


def _strip_legacy_transition_keys(transition_data):
    if not isinstance(transition_data, dict):
        return transition_data
    if not any(key in transition_data for key in _LEGACY_TRANSITION_KEYS):
        return transition_data
    return {k: v for k, v in transition_data.items() if k not in _LEGACY_TRANSITION_KEYS}


def _make_signals_class():
    """Create the SceneAnimationSignals class only when Qt is available."""
    from Qt.QtCore import QObject, Signal as pyqtSignal

    class SceneAnimationSignals(QObject):
        """Signal emitter for SceneAnimation to avoid metaclass conflicts"""

        time_changed = pyqtSignal(float)  # Current playback time
        duration_changed = pyqtSignal(float)  # Animation duration changed
        playback_started = pyqtSignal()
        playback_stopped = pyqtSignal()
        recording_started = pyqtSignal()
        recording_stopped = pyqtSignal()
        timeline_cleared = pyqtSignal()  # Scenes and actions cleared by command

    return SceneAnimationSignals


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

        # Animation state
        self.duration = self.DEFAULT_DURATION
        self.scenes = []  # List of (time, scene_name, transition_data) tuples
        # transition_data = {'type': 'linear'}
        self.action_segments = []  # List of (start_time, end_time, action_name) tuples for rock/roll
        self.current_time = 0.0
        self.is_playing = False
        self.is_recording = False

        # Playback state
        self.fps = fps
        self.start_time = 0.0
        self.reverse = False

        # Qt objects are only available in GUI mode
        self._is_gui = hasattr(session, 'ui') and session.ui.is_gui
        if self._is_gui:
            from Qt.QtCore import QTimer
            SignalsClass = _make_signals_class()
            self.signals = SignalsClass()
            self.playback_timer = QTimer()
            self.playback_timer.timeout.connect(self._advance_playback)
        else:
            self.signals = None
            self.playback_timer = None

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

    def move_scene_at_time(self, old_time: float, new_time: float):
        """Move the scene entry at old_time to new_time, preserving its
        scene name and transition data. Any existing entry at new_time is
        replaced."""
        entry = None
        for t, s, td in self.scenes:
            if t == old_time:
                entry = (s, td)
                break
        if entry is None:
            return False
        self.scenes = [
            (t, s, td) for t, s, td in self.scenes
            if t != old_time and t != new_time
        ]
        self.scenes.append((new_time, entry[0], entry[1]))
        self.scenes.sort(key=lambda x: x[0])
        return True

    def set_scene_transition(self, time: float, transition_data: dict):
        """Replace the transition data for the scene entry at the given time."""
        for i, (t, s, td) in enumerate(self.scenes):
            if t == time:
                self.scenes[i] = (t, s, dict(transition_data))
                return True
        return False

    def set_duration(self, duration: float):
        """Set the total duration of the animation"""
        if duration <= 0:
            #self.logger.warning("Duration must be positive")
            return False

        self.duration = duration

        # Remove any scenes beyond the new duration
        self.scenes = [(t, s, td) for t, s, td in self.scenes if t <= duration]

        if self.signals:
            self.signals.duration_changed.emit(duration)
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
            return

        self.current_time = time

        # Find the appropriate scene or transition
        scene1, scene2, fraction = self._get_interpolation_at_time(time)
        active_action_segment = self._get_active_action_segment_index(time)
        previous_active_action_segment = getattr(
            self, "_last_active_action_segment", None
        )

        # Check if the scene state changed from what we're currently displaying
        scene_changed = True
        if hasattr(self, "_last_scene_state"):
            if self._last_scene_state == (scene1, scene2, fraction):
                scene_changed = False
            elif self._last_scene_state[:2] != (scene1, scene2):
                self._last_action_angle = 0.0

        base_state_restored = False
        if not self.scenes and active_action_segment is None:
            self._ensure_action_only_base_state()
            self._restore_action_only_base_state()
            base_state_restored = True
        elif scene_changed or previous_active_action_segment != active_action_segment:
            self._restore_preview_base_state(scene1, scene2, fraction, time)
            base_state_restored = True

        if base_state_restored:
            self._reset_action_tracking()
            self._last_scene_state = (scene1, scene2, fraction)

        # Apply action segments after scene restore so they layer on top of
        # the base scene state. During sequential playback the scene state
        # won't change every frame, but actions still need to update.
        self._apply_action_segments(time)
        self._apply_trajectory_at_time(scene1, scene2, fraction)
        self._last_active_action_segment = active_action_segment

        # Notify the UI so the playhead tracks the previewed time.
        # Skip during playback — _advance_playback emits time_changed itself,
        # and re-emitting here would create a signal loop.
        if self.signals and not self.is_playing:
            self.signals.time_changed.emit(time)

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
        if self.signals:
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
            if self.playback_timer:
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
        if self.signals:
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
        if self.signals:
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
        if self.playback_timer:
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
        if self.signals:
            self.signals.playback_stopped.emit()
        #self.logger.status("Stopped animation")

    def set_fps(self, fps: int):
        """Update FPS and restart timer if playing"""
        self.fps = fps
        if self.is_playing and self.playback_timer:
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
            if self.signals:
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
            if self.signals:
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

    def _get_active_action_segment_index(self, time: float):
        """Return the active action segment index for a preview time."""
        for index, segment_data in enumerate(self.action_segments):
            start_time, end_time, _action_name = segment_data[:3]
            if start_time <= time <= end_time:
                return index
        return None

    def _restore_preview_base_state(
        self, scene1: str, scene2: str, fraction: float, time: float
    ):
        """Restore the base state for the current preview time before actions."""
        if not self.scenes:
            self._ensure_action_only_base_state()
            self._restore_action_only_base_state()
            return

        if scene1 == scene2:
            if scene1:
                self.session.scenes.restore_scene(scene1)
            return

        if scene1 and scene2:
            scene2_data = self._get_scene_transition_data(scene2)
            self.session.scenes.interpolate_scenes(scene1, scene2, fraction)
            action = scene2_data.get("action") if scene2_data else None
            if action:
                self._apply_action(action, fraction)

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
                        # Precess: figure-8 wobble matching ChimeraX's wobble command
                        # Primary rotation on main axis at frequency f,
                        # secondary rotation on perpendicular axis at 2f
                        wobble_aspect = config.get("wobble_aspect", 0.3)

                        # Compute wobble position as compound rotation, same as
                        # Turner._wobble_position in std_commands/turn.py
                        amax = 0.5 * angle
                        f0 = fraction * count  # Current normalized position
                        # We need the previous frame's position too for incremental motion
                        if not hasattr(self, '_wobble_last_fraction'):
                            self._wobble_last_fraction = {}
                        f_prev = self._wobble_last_fraction.get(segment_key, 0.0)
                        self._wobble_last_fraction[segment_key] = f0

                        # Get axis vectors
                        axis_map = {'x': (1, 0, 0), 'y': (0, 1, 0), 'z': (0, 0, 1)}
                        axis_vec = axis_map.get(axis, (0, 1, 0))

                        # Compute wobble axis from camera view direction cross primary axis
                        from chimerax.geometry import cross_product, normalize_vector, rotation
                        camera = self.session.view.camera
                        wobble_axis_vec = normalize_vector(
                            cross_product(camera.view_direction(), axis_vec)
                        )

                        # Compute wobble positions at previous and current fraction
                        def wobble_pos(f):
                            a = math.sin(2 * math.pi * f) * amax
                            wa = math.sin(4 * math.pi * f) * amax * wobble_aspect
                            r = rotation(axis_vec, a, center)
                            rw = rotation(wobble_axis_vec, wa, center)
                            return rw * r

                        w_prev = wobble_pos(f_prev)
                        w_curr = wobble_pos(f0)

                        # Incremental rotation: for camera motion, use w_prev * w_curr.inverse()
                        incremental = w_prev * w_curr.inverse()
                        camera.position = incremental * camera.position

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
        if hasattr(self, '_wobble_last_fraction'):
            self._wobble_last_fraction.clear()

    def _apply_trajectory_at_time(self, scene1_name, scene2_name, fraction):
        """Drive a morph trajectory's active coordset between two scenes.

        Reads each scene's saved ``active_coordset_id`` for the auto-picked
        morph trajectory and linearly interpolates between them at
        ``fraction``. At a steady scene (``scene1_name == scene2_name``)
        snaps to that scene's captured frame.
        """
        if not scene1_name or not scene2_name:
            return

        from .trajectory import find_morph_trajectory, interpolate_trajectory_ids
        traj = find_morph_trajectory(self.session)
        if traj is None:
            return

        s1 = self.session.scenes.get_scene(scene1_name)
        s2 = self.session.scenes.get_scene(scene2_name)
        if s1 is None or s2 is None:
            return

        id_a = self._get_scene_coordset_id(s1, traj)
        id_b = self._get_scene_coordset_id(s2, traj)
        if id_a is None or id_b is None:
            return

        interpolate_trajectory_ids(traj, id_a, id_b, fraction)

    def _get_scene_coordset_id(self, scene, traj):
        """Return the saved active_coordset_id for ``traj`` in ``scene``, or None.

        ``AtomicStructure.take_snapshot`` wraps the ``Structure`` SCENE dict
        under ``'structure state'``; plain ``Structure`` puts it at the top.
        """
        info = scene.scene_models.get(traj)
        if info is None:
            return None
        _, scene_data = info
        if not isinstance(scene_data, dict):
            return None
        inner = scene_data.get('structure state', scene_data)
        return inner.get('structure', {}).get('active_coordset_id')

    def _get_trajectory_fraction(self, time: float) -> float:
        """Compute a global trajectory fraction in [0, 1] across all scenes.

        Within each scene-to-scene segment, applies that segment's easing
        function so the morph progresses with the same feel as the scene
        transition. Before the first scene returns 0; after the last, 1.
        """
        sorted_scenes = sorted(self.scenes, key=lambda x: x[0])
        n = len(sorted_scenes)
        if n < 2:
            return 0.0
        if time <= sorted_scenes[0][0]:
            return 0.0
        if time >= sorted_scenes[-1][0]:
            return 1.0

        for i in range(n - 1):
            t1, _, _ = sorted_scenes[i]
            t2, _, transition2 = sorted_scenes[i + 1]
            if t1 <= time <= t2:
                seg_dur = t2 - t1
                local = (time - t1) / seg_dur if seg_dur > 0 else 0.0
                local = max(0.0, min(1.0, local))
                transition_type = (transition2 or {}).get("type", "linear")
                easing = TRANSITION_TYPES.get(transition_type, EasingFunctions.linear)
                local_eased = easing(local)
                base = i / (n - 1)
                step = 1.0 / (n - 1)
                return base + local_eased * step
        return 1.0

    def _reset_action_tracking(self):
        """Reset action segment tracking state.

        Called after a scene restore or interpolation resets the camera to a
        base state, so that subsequent action application computes its effect
        as an absolute offset from zero rather than an incremental delta from
        a stale previous value.
        """
        if hasattr(self, '_segment_angles'):
            self._segment_angles.clear()
        if hasattr(self, '_wobble_last_fraction'):
            self._wobble_last_fraction.clear()
        if hasattr(self, '_last_action_angle'):
            self._last_action_angle = 0.0

    def _ensure_action_only_base_state(self):
        """Capture the current view as the base state for action-only previews."""
        if hasattr(self, "_action_only_base_camera_position"):
            return

        view = self.session.view
        self._action_only_base_camera_position = view.camera.position
        self._action_only_center_of_rotation = getattr(view, "center_of_rotation", None)
        self._action_only_center_of_rotation_method = getattr(
            view, "center_of_rotation_method", None
        )

    def _restore_action_only_base_state(self):
        """Restore the captured base state for action-only previews."""
        if not hasattr(self, "_action_only_base_camera_position"):
            return

        view = self.session.view
        view.camera.position = self._action_only_base_camera_position

        if hasattr(self, "_action_only_center_of_rotation"):
            view.center_of_rotation = self._action_only_center_of_rotation

        if (
            hasattr(self, "_action_only_center_of_rotation_method")
            and self._action_only_center_of_rotation_method is not None
        ):
            view.center_of_rotation_method = self._action_only_center_of_rotation_method

    def get_scene_list(self) -> List[Tuple[float, str]]:
        """Get list of all scenes with their times (for compatibility)"""
        return [(t, s) for t, s, _ in self.scenes]

    def get_scene_list_with_transitions(self) -> List[Tuple[float, str, dict]]:
        """Get list of all scenes with their times and transition data"""
        return self.scenes.copy()

    def clear_all_scenes(self):
        """Remove all scenes from the animation"""
        self.scenes = []

    def clear_timeline(self):
        """Remove all scenes and action segments and notify listeners.

        This is the public entry point for the ``animations clear`` command.
        ``clear_all_scenes`` is the lower-level primitive used by
        ``reset_state`` during session resets, where emitting a
        ``timeline_cleared`` signal would be inappropriate.
        """
        self.clear_all_scenes()
        self.clear_action_segments()
        if self.signals:
            self.signals.timeline_cleared.emit()

    # ------------------------------------------------------------------
    # Action segment management
    # ------------------------------------------------------------------

    def add_action_segment(self, start_time, end_time, action_name, config=None):
        """Add an action segment to the animation."""
        if config is None:
            config = ACTION_DEFAULTS.get(action_name, {}).copy()
        self.action_segments.append((start_time, end_time, action_name, config))
        self.action_segments.sort(key=lambda x: x[0])

    def remove_action_segment(self, index):
        """Remove an action segment by index."""
        if 0 <= index < len(self.action_segments):
            del self.action_segments[index]

    def update_action_segment(self, index, start_time, end_time, action_name, config):
        """Replace an action segment at the given index."""
        if 0 <= index < len(self.action_segments):
            self.action_segments[index] = (start_time, end_time, action_name, config)

    def clear_action_segments(self):
        """Remove all action segments from the animation."""
        self.action_segments = []

    def take_snapshot(self, session, flags):
        """Save state for session snapshots"""
        return {
            "version": self.version,
            "duration": self.duration,
            "scenes": self.scenes,
            "action_segments": self.action_segments,
            "current_time": self.current_time,
        }

    def restore_from_data(self, data):
        """Restore state from snapshot data.

        Scene validation is deferred because the SceneManager may not have
        restored its scenes yet when this method runs during session restore.
        Call validate_scenes() after the full session has been restored.
        """
        if data.get("version", 0) != self.version:
            return

        self.duration = data.get("duration", self.DEFAULT_DURATION)
        self.scenes = [
            (time, scene_name, _strip_legacy_transition_keys(transition_data))
            for time, scene_name, transition_data in data.get("scenes", [])
        ]
        self.action_segments = data.get("action_segments", [])
        self.current_time = data.get("current_time", 0.0)

    def validate_scenes(self):
        """Remove any scenes whose backing Scene no longer exists.

        Should be called after the full session has been restored so
        that session.scenes is fully populated.
        """
        valid_scenes = []
        for time, scene_name, transition_data in self.scenes:
            if self.session.scenes.get_scene(scene_name):
                valid_scenes.append((time, scene_name, transition_data))
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
