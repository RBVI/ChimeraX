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

        # Use the session-registered SceneAnimation state manager
        self.scene_animation = session.get_state_manager("scene animations")
        self.scene_animation.set_fps(fps)

        # Connect signals
        self._connect_widget_signals()
        self._connect_animation_signals()
        self._scene_deleted_handler = self._connect_scene_manager_signals()

    def _connect_widget_signals(self):
        """Connect signals from the widget to controller handlers."""
        self.widget.scene_added.connect(self._on_scene_added)
        self.widget.scene_removed.connect(self._on_scene_removed)
        self.widget.scene_moved.connect(self._on_scene_moved)
        self.widget.scene_selected.connect(self._on_scene_selected)
        self.widget.scene_transition_changed.connect(self._on_scene_transition_changed)
        self.widget.action_added.connect(self._on_action_added)
        self.widget.action_removed.connect(self._on_action_removed)
        self.widget.action_updated.connect(self._on_action_updated)
        self.widget.time_changed.connect(self._on_time_changed)
        self.widget.play_requested.connect(self._on_play_requested)
        self.widget.pause_requested.connect(self._on_pause_requested)
        self.widget.record_requested.connect(self._on_record_requested)
        self.widget.duration_changed.connect(self._on_duration_changed)
        self.widget.reset_requested.connect(self._on_reset_requested)

    def _connect_animation_signals(self):
        """Connect signals from SceneAnimation to update the widget."""
        self.scene_animation.signals.time_changed.connect(self._on_animation_time_changed)
        self.scene_animation.signals.duration_changed.connect(self._on_animation_duration_changed)
        self.scene_animation.signals.playback_started.connect(self._on_animation_started)
        self.scene_animation.signals.playback_stopped.connect(self._on_animation_stopped)
        self.scene_animation.signals.recording_started.connect(self._on_recording_started)
        self.scene_animation.signals.recording_stopped.connect(self._on_recording_stopped)
        self.scene_animation.signals.timeline_cleared.connect(self._on_animation_timeline_cleared)

    def _connect_scene_manager_signals(self):
        """Listen for scene deletions in the scenes manager so the timeline
        drops any markers referencing a scene that no longer exists."""
        from chimerax.scenes.triggers import add_handler, DELETED
        return add_handler(DELETED, self._on_scene_manager_deleted)

    def _on_scene_manager_deleted(self, _trigger_name, scene_name):
        """Remove all timeline entries for a scene that was deleted from the
        scenes manager (e.g. via the Scenes GUI delete button)."""
        if not self.scene_animation.remove_scene(scene_name):
            return
        self.widget.timeline_scene.remove_scene_marker(scene_name)

    # -------------------------------------------------------------------------
    # Widget -> Controller handlers
    # -------------------------------------------------------------------------

    def _on_scene_added(self, scene_name: str, time: float = None):
        """Handle scene addition from the widget."""
        if time is None:
            time = self.widget.timeline_controls.current_time

        self.scene_animation.add_scene_at_time(scene_name, time)

    def _on_scene_removed(self, scene_name: str, time: float = None):
        """Handle scene removal from the widget."""
        if time is not None:
            success = self.scene_animation.remove_scene_at_time(time)
        else:
            success = self.scene_animation.remove_scene(scene_name)
        if success:
            self.widget.timeline_scene.remove_scene_marker(scene_name, time)

    def _on_scene_moved(self, scene_name: str, old_time: float, new_time: float):
        """Handle scene marker moved on timeline."""
        self.scene_animation.move_scene_at_time(old_time, new_time)

    def _on_scene_selected(self, scene_name: str):
        """Handle scene selection - restore the scene."""
        if self.session and self.session.scenes.get_scene(scene_name):
            self.session.scenes.restore_scene(scene_name)

    def _on_scene_transition_changed(self, time: float, transition_data: dict):
        """Handle transition type change from the context menu."""
        self.scene_animation.set_scene_transition(time, transition_data)

    def _on_action_added(self, start_time: float, end_time: float,
                         action_name: str, config: dict):
        """Handle action segment added from the widget."""
        self.scene_animation.add_action_segment(start_time, end_time,
                                                action_name, config)

    def _on_action_removed(self, index: int):
        """Handle action segment removed from the widget."""
        self.scene_animation.remove_action_segment(index)

    def _on_action_updated(self, index: int, start_time: float,
                           end_time: float, action_name: str, config: dict):
        """Handle action segment moved, resized, or reconfigured."""
        self.scene_animation.update_action_segment(index, start_time,
                                                   end_time, action_name, config)

    def _on_time_changed(self, time: float):
        """Handle time scrubber changes for preview."""
        self.scene_animation.preview_at_time(time)

    def _on_play_requested(self):
        """Handle play button pressed."""
        current_time = self.widget.timeline_controls.current_time
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

    def _on_animation_duration_changed(self, duration: float):
        """Handle duration changes from the animation (e.g. via command)."""
        self.widget.timeline_scene.duration = duration
        self.widget.timeline_scene.update()
        self.widget.timeline_controls.set_duration(duration)

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

    def _on_animation_timeline_cleared(self):
        """Handle the animation timeline being cleared via command.

        Drops the widget-side scene markers and action segments so the
        GUI matches the now-empty SceneAnimation state.
        """
        timeline_scene = self.widget.timeline_scene
        timeline_scene.scene_markers = []
        timeline_scene.action_segments = []
        timeline_scene.selected_scene_marker_id = None
        timeline_scene.dragging_scene_marker_id = None
        timeline_scene.potential_drag_scene_marker_id = None
        timeline_scene.selected_action_segment = None
        timeline_scene.dragging_action_segment = None
        timeline_scene.resizing_action_segment = None
        timeline_scene.update()

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _get_movie_save_path_and_options(self):
        """Get save path and recording options using dialog."""
        # Import here to avoid circular imports
        from .editor_widget import MovieRecordingDialog
        dialog = MovieRecordingDialog(self.session, parent=self.widget)
        if dialog.exec():
            return dialog.get_save_path(), dialog.get_resolution()
        return None, None

    def cleanup(self):
        """Clean up resources when the tool is closed.

        SceneAnimation is a session-level state manager that outlives the
        tool, so its signal connections must be torn down explicitly.
        Otherwise this controller (and its deleted Qt widgets) will keep
        receiving callbacks after the tool is gone.
        """
        self.scene_animation.stop_playing()
        signals = self.scene_animation.signals
        if signals is not None:
            signals.time_changed.disconnect(self._on_animation_time_changed)
            signals.duration_changed.disconnect(self._on_animation_duration_changed)
            signals.playback_started.disconnect(self._on_animation_started)
            signals.playback_stopped.disconnect(self._on_animation_stopped)
            signals.recording_started.disconnect(self._on_recording_started)
            signals.recording_stopped.disconnect(self._on_recording_stopped)
            signals.timeline_cleared.disconnect(self._on_animation_timeline_cleared)

        if self._scene_deleted_handler is not None:
            from chimerax.scenes.triggers import remove_handler
            remove_handler(self._scene_deleted_handler)
            self._scene_deleted_handler = None
