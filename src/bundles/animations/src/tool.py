"""
This module defines the AnimationsTool class for managing animations in ChimeraX.

Classes:
    AnimationsTool: A tool instance for managing animations, including adding, editing, deleting keyframes,
        and handling triggers.
"""

from Qt.QtWidgets import QVBoxLayout, QHBoxLayout, QStackedWidget, QPushButton, QButtonGroup

from chimerax.core.commands import run
from chimerax.core.tools import ToolInstance

from chimerax.ui.open_save import SaveDialog

from chimerax.animations.triggers import (add_handler, KF_EDIT, PREVIEW, PLAY, KF_ADD, KF_DELETE, RECORD, STOP_PLAYING, INSERT_TIME, REMOVE_TIME, remove_handler, STOP_RECORDING)
from chimerax.animations.kf_editor_two import KeyframeEditorWidget
from chimerax.animations.scene_timeline import SceneTimelineWidget
from chimerax.animations.scene_animation import SceneAnimation


class AnimationsTool(ToolInstance):
    """
    A tool instance for managing animations, including adding, editing, deleting keyframes, and handling triggers.

    Attributes:
        SESSION_ENDURING (bool): Whether the instance persists when the session closes.
        SESSION_SAVE (bool): Whether the instance is saved/restored in sessions.
        handlers (list): List of handlers for triggers.
        display_name (str): Name displayed on the title bar.
        animation_mgr: Reference to the animation manager.
        tool_window: Main tool window for the UI.
    """

    SESSION_ENDURING = False  # Does this instance persist when session closes
    SESSION_SAVE = True  # We do save/restore in sessions

    def __init__(self, session, tool_name):
        """
        Initialize the AnimationsTool.

        Args:
            session: The current session.
            tool_name (str): The name of the tool.
        """

        super().__init__(session, tool_name)

        self.handlers = []

        # Set name displayed on title bar (defaults to tool_name)
        # Must be after the superclass init, which would override it.
        self.display_name = "Animations"

        # Store a reference to the animation manager
        self.animation_mgr = self.session.get_state_manager("animations")

        # Get or create scene animation manager
        self.scene_animation = self._get_scene_animation_manager()

        from chimerax.ui import MainToolWindow
        self.tool_window = MainToolWindow(self)

        # Build UI with mode switching
        self.build_ui()

        # Register handlers for the triggers (only for keyframe mode)
        self.handlers.append(add_handler(PREVIEW, lambda trigger_name, time: run(self.session, f"animations preview {time}")))
        self.handlers.append(add_handler(KF_EDIT, lambda trigger_name, data: run(self.session, f"animations keyframe edit {data[0]} time {data[1]}")))
        self.handlers.append(add_handler(PLAY, lambda trigger_name, data: run(self.session, f"animations play start {data[0]} reverse {data[1]}")))
        self.handlers.append(add_handler(KF_ADD, lambda trigger_name, time: self.add_keyframe(time)))
        self.handlers.append(add_handler(KF_DELETE, lambda trigger_name, kf_name: run(self.session, f"animations keyframe delete {kf_name}")))
        self.handlers.append(add_handler(RECORD, lambda trigger_name, data: self.record()))
        self.handlers.append(add_handler(STOP_RECORDING, lambda trigger_name, data: run(self.session, "animations stopRecording")))
        self.handlers.append(add_handler(STOP_PLAYING, lambda trigger_name, data: run(self.session, "animations stop")))
        self.handlers.append(add_handler(INSERT_TIME, lambda trigger_name, data: run(self.session, f"animations insertTime {data[0]} {data[1]}")))
        self.handlers.append(add_handler(REMOVE_TIME, lambda trigger_name, data: run(self.session, f"animations removeTime {data[0]} {data[1]}")))

        self.tool_window.manage("bottom")

    def build_ui(self):
        """
        Build the user interface for the tool.
        """

        main_vbox_layout = QVBoxLayout()

        # Mode switching controls
        mode_controls_layout = QHBoxLayout()

        self.mode_button_group = QButtonGroup()

        self.keyframe_mode_btn = QPushButton("Keyframe Mode")
        self.keyframe_mode_btn.setCheckable(True)
        self.keyframe_mode_btn.setChecked(True)  # Default to keyframe mode
        self.keyframe_mode_btn.setStyleSheet("""
            QPushButton:checked {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }
        """)

        self.scene_mode_btn = QPushButton("Scene Mode")
        self.scene_mode_btn.setCheckable(True)
        self.scene_mode_btn.setStyleSheet("""
            QPushButton:checked {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
            }
        """)

        self.mode_button_group.addButton(self.keyframe_mode_btn, 0)
        self.mode_button_group.addButton(self.scene_mode_btn, 1)
        self.mode_button_group.buttonClicked.connect(self.switch_mode)

        mode_controls_layout.addWidget(self.keyframe_mode_btn)
        mode_controls_layout.addWidget(self.scene_mode_btn)
        mode_controls_layout.addStretch()

        main_vbox_layout.addLayout(mode_controls_layout)

        # Stacked widget for different modes
        self.stacked_widget = QStackedWidget()

        # Keyframe editor mode (original)
        self.kf_editor_widget = KeyframeEditorWidget(self.session)
        self.stacked_widget.addWidget(self.kf_editor_widget)

        # Scene timeline mode (new)
        self.scene_timeline_widget = SceneTimelineWidget(self.session)
        self.stacked_widget.addWidget(self.scene_timeline_widget)

        # Connect scene timeline signals
        self.scene_timeline_widget.scene_added.connect(self.on_scene_added)
        self.scene_timeline_widget.scene_removed.connect(self.on_scene_removed)
        self.scene_timeline_widget.scene_selected.connect(self.on_scene_selected)
        self.scene_timeline_widget.time_changed.connect(self.on_scene_time_changed)

        # Connect timeline drag and drop
        self.scene_timeline_widget.timeline_scene.scene_dropped.connect(self.on_scene_dropped)

        main_vbox_layout.addWidget(self.stacked_widget)

        self.tool_window.ui_area.setLayout(main_vbox_layout)

    def _get_scene_animation_manager(self):
        """Get or create the scene animation manager"""
        if not hasattr(self.session, '_scene_animation_manager'):
            self.session._scene_animation_manager = SceneAnimation(self.session)
        return self.session._scene_animation_manager

    def switch_mode(self, button):
        """Switch between keyframe and scene animation modes"""
        if button == self.keyframe_mode_btn:
            self.stacked_widget.setCurrentIndex(0)
            self.session.logger.info("Switched to Keyframe Mode")
        elif button == self.scene_mode_btn:
            self.stacked_widget.setCurrentIndex(1)
            self.session.logger.info("Switched to Scene Mode")

    def on_scene_added(self, scene_name, time=None):
        """Handle scene addition in scene mode"""
        if time is None:
            time = self.scene_timeline_widget.timeline_controls.current_time

        success = self.scene_animation.add_scene_at_time(scene_name, time)

        if success:
            self.scene_timeline_widget.add_scene_marker(time, scene_name)
            self.session.logger.info(f"Added scene '{scene_name}' to animation at {time:.2f}s")

    def on_scene_removed(self, scene_name):
        """Handle scene removal in scene mode"""
        success = self.scene_animation.remove_scene(scene_name)

        if success:
            self.scene_timeline_widget.timeline_scene.remove_scene_marker(scene_name)
            self.session.logger.info(f"Removed scene '{scene_name}' from animation")

    def on_scene_selected(self, scene_name):
        """Handle scene selection in scene mode"""
        if self.session.scenes.get_scene(scene_name):
            self.session.scenes.restore_scene(scene_name)
            self.session.logger.info(f"Restored scene '{scene_name}'")

    def on_scene_time_changed(self, time):
        """Handle timeline time change in scene mode"""
        self.scene_animation.preview_at_time(time)
        self.scene_timeline_widget.set_current_time(time)

    def on_scene_dropped(self, scene_name, time):
        """Handle scene dropped onto timeline"""
        self.on_scene_added(scene_name, time)

    def add_keyframe(self, time):
        """
        Add a keyframe at the specified time.

        Args:
            time (int | float): The time in seconds to add the keyframe.
        """
        base_name = "keyframe_"
        id = 0
        while any(kf.get_name() == f"{base_name}{id}" for kf in self.animation_mgr.get_keyframes()):
            id += 1
        kf_name = f"{base_name}{id}"
        run(self.session, f"animations keyframe add {kf_name} time {time}")

    def record(self):
        """
        Start recording the animation.
        """
        save_path = self.get_save_path()
        if save_path is None:
            return
        run(self.session, f"animations record {save_path}")

    def get_save_path(self):
        save_dialog = SaveDialog(self.session, parent=self.tool_window.ui_area)
        save_dialog.setNameFilter("Video Files (*.mp4 *.mov *.avi *.wmv)")
        if save_dialog.exec():
            file_path = save_dialog.selectedFiles()[0]
            return file_path
        return None

    def delete(self):
        """
        Override from super. Remove the trigger handlers before the tool is deleted.
        """
        for handler in self.handlers:
            remove_handler(handler)
        if self.kf_editor_widget is not None:
            self.kf_editor_widget.remove_handlers()

        # Stop any scene animation playback
        if hasattr(self, 'scene_animation') and self.scene_animation:
            self.scene_animation.stop_playing()

        super().delete()

    def take_snapshot(self, session, flags):
        return {
            'version': 1
        }

    @classmethod
    def restore_snapshot(class_obj, session, data):
        inst = class_obj(session, "Animations")
        return inst
