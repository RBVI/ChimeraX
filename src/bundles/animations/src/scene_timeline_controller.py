# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2025 Regents of the University of California. All rights reserved.
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

"""
Controller for the Scene Timeline mode.

This module provides the SceneTimelineController class which manages the
interaction between the SceneTimelineWidget (UI) and SceneAnimation (model).
It encapsulates all the signal wiring and event handling that was previously
spread across KeyframeEditorWidget.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .scene_timeline import SceneTimelineWidget

from .scene_animation import SceneAnimation


class SceneTimelineController:
    """
    Controller for the Scene Timeline mode.

    This class owns the SceneAnimation instance and handles all communication
    between the SceneTimelineWidget and the animation system. It replaces the
    tangled signal connections that were previously in KeyframeEditorWidget.

    Data flow is unidirectional:
        Widget -> Controller -> SceneAnimation -> Controller -> Widget
    """

    def __init__(self, session, widget: "SceneTimelineWidget", fps: int = 60):
        """
        Initialize the controller.

        Parameters
        ----------
        session : ChimeraX session
            The ChimeraX session object.
        widget : SceneTimelineWidget
            The scene timeline widget to control.
        fps : int
            Frames per second for animation playback.
        """
        self.session = session
        self.widget = widget
        self.fps = fps

        # Create our own SceneAnimation - NOT stored on session
        self.scene_animation = SceneAnimation(session, fps=fps)

        # Connect signals
        self._connect_widget_signals()
        self._connect_animation_signals()

    def _connect_widget_signals(self):
        """Connect signals from the widget to controller handlers."""
        self.widget.scene_added.connect(self._on_scene_added)
        self.widget.scene_removed.connect(self._on_scene_removed)
        self.widget.scene_moved.connect(self._on_scene_moved)
        self.widget.scene_selected.connect(self._on_scene_selected)
        self.widget.time_changed.connect(self._on_time_changed)
        self.widget.play_requested.connect(self._on_play_requested)
        self.widget.pause_requested.connect(self._on_pause_requested)
        self.widget.record_requested.connect(self._on_record_requested)
        self.widget.duration_changed.connect(self._on_duration_changed)
        self.widget.reset_requested.connect(self._on_reset_requested)

    def _connect_animation_signals(self):
        """Connect signals from SceneAnimation to update the widget."""
        self.scene_animation.signals.time_changed.connect(self._on_animation_time_changed)
        self.scene_animation.signals.playback_started.connect(self._on_animation_started)
        self.scene_animation.signals.playback_stopped.connect(self._on_animation_stopped)
        self.scene_animation.signals.recording_started.connect(self._on_recording_started)
        self.scene_animation.signals.recording_stopped.connect(self._on_recording_stopped)

    # -------------------------------------------------------------------------
    # Widget -> Controller handlers
    # -------------------------------------------------------------------------

    def _on_scene_added(self, scene_name: str, time: float = None):
        """Handle scene addition from the widget."""
        if time is None:
            time = self.widget.timeline_controls.current_time

        success = self.scene_animation.add_scene_at_time(scene_name, time)
        if success:
            self.widget.add_scene_marker(time, scene_name)

    def _on_scene_removed(self, scene_name: str):
        """Handle scene removal from the widget."""
        success = self.scene_animation.remove_scene(scene_name)
        if success:
            self.widget.timeline_scene.remove_scene_marker(scene_name)

    def _on_scene_moved(self, scene_name: str, old_time: float, new_time: float):
        """Handle scene marker moved on timeline."""
        self.scene_animation.remove_scene_at_time(old_time)
        self.scene_animation.add_scene_at_time(scene_name, new_time)

    def _on_scene_selected(self, scene_name: str):
        """Handle scene selection - restore the scene."""
        if self.session and self.session.scenes.get_scene(scene_name):
            self.session.scenes.restore_scene(scene_name)

    def _on_time_changed(self, time: float):
        """Handle time scrubber changes for preview."""
        self._sync_to_animation()
        self.scene_animation.preview_at_time(time)

    def _on_play_requested(self):
        """Handle play button pressed."""
        current_time = self.widget.timeline_controls.current_time
        self._sync_to_animation()
        self.scene_animation.play(start_time=current_time)

    def _on_pause_requested(self):
        """Handle pause button pressed."""
        self.scene_animation.stop_playing()

    def _on_record_requested(self):
        """Handle record button pressed."""
        if self.scene_animation.is_recording:
            self.scene_animation.stop_playing()
            return

        if not self.scene_animation.scenes:
            return

        save_path, resolution = self._get_movie_save_path_and_options()
        if save_path:
            self.scene_animation.record(save_path, resolution=resolution)

    def _on_duration_changed(self, new_duration: float):
        """Handle duration change from zoom buttons."""
        self.scene_animation.set_duration(new_duration)
        self.widget.timeline_controls.set_duration(new_duration)

    def _on_reset_requested(self):
        """Handle reset button pressed."""
        self.scene_animation.stop_playing()
        self.scene_animation.preview_at_time(0.0)

    # -------------------------------------------------------------------------
    # Animation -> Controller -> Widget handlers
    # -------------------------------------------------------------------------

    def _on_animation_time_changed(self, time: float):
        """Handle time updates from animation during playback."""
        self.widget.set_current_time(time)

    def _on_animation_started(self):
        """Handle animation playback started."""
        self.widget.set_playing_state(True)

    def _on_animation_stopped(self):
        """Handle animation playback stopped."""
        self.widget.set_playing_state(False)

    def _on_recording_started(self):
        """Handle recording started."""
        self.widget.set_recording_state(True)

    def _on_recording_stopped(self):
        """Handle recording stopped."""
        self.widget.set_recording_state(False)

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _sync_to_animation(self):
        """Sync scene markers and action segments from widget to animation."""
        self.scene_animation.clear_all_scenes()

        duration = self.widget.timeline_scene.duration
        self.scene_animation.set_duration(duration)

        for marker_data in self.widget.timeline_scene.scene_markers:
            if len(marker_data) >= 4:
                time, scene_name, pixmap, transition_data = marker_data
                self.scene_animation.add_scene_at_time(
                    scene_name, time,
                    transition_data.get('type', 'linear'),
                    transition_data.get('fade_models', False)
                )
            elif len(marker_data) >= 2:
                time, scene_name = marker_data[:2]
                self.scene_animation.add_scene_at_time(scene_name, time)

        self.scene_animation.action_segments = list(
            self.widget.timeline_scene.action_segments
        )

    def _get_movie_save_path_and_options(self):
        """Get save path and recording options using dialog."""
        # Import here to avoid circular imports
        from .kf_editor_two import MovieRecordingDialog
        dialog = MovieRecordingDialog(self.session, parent=self.widget)
        if dialog.exec():
            return dialog.get_save_path(), dialog.get_resolution()
        return None, None

    def cleanup(self):
        """Clean up resources when the tool is closed."""
        self.scene_animation.stop_playing()
