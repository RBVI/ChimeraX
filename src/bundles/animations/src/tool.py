"""
This module defines the AnimationsTool class for managing animations in ChimeraX.

Classes:
    AnimationsTool: A tool instance for managing animations, including adding, editing, deleting keyframes,
        and handling triggers.
"""

from Qt.QtWidgets import QVBoxLayout

from chimerax.core.commands import run
from chimerax.core.tools import ToolInstance

from chimerax.ui.open_save import SaveDialog

from chimerax.animations.triggers import (add_handler, KF_EDIT, PREVIEW, PLAY, KF_ADD, KF_DELETE, RECORD, STOP_PLAYING, INSERT_TIME, REMOVE_TIME, remove_handler, STOP_RECORDING)
from chimerax.animations.kf_editor_two import KeyframeEditorWidget


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

        # Use the enhanced KeyframeEditorWidget that includes dual-mode support
        self.kf_editor_widget = KeyframeEditorWidget(self.session)

        # Hide mode toggle buttons to reclaim vertical space
        self._hide_mode_buttons()

        # Initialize mode based on settings
        self._initialize_mode_from_settings()

        # Set up settings change monitoring
        self._setup_settings_monitoring()

        main_vbox_layout.addWidget(self.kf_editor_widget)

        self.tool_window.ui_area.setLayout(main_vbox_layout)

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

    def _hide_mode_buttons(self):
        """Hide the mode toggle buttons to reclaim vertical space."""
        self.kf_editor_widget.keyframe_mode_btn.hide()
        self.kf_editor_widget.scene_mode_btn.hide()

    def _initialize_mode_from_settings(self):
        """Initialize the current mode based on the saved settings."""
        from .settings import get_settings
        settings = get_settings(self.session)
        mode = settings.animation_mode

        if mode == 'keyframe':
            self.kf_editor_widget.keyframe_mode_btn.setChecked(True)
            self.kf_editor_widget.switch_mode(self.kf_editor_widget.keyframe_mode_btn)
        else:  # scene mode
            self.kf_editor_widget.scene_mode_btn.setChecked(True)
            self.kf_editor_widget.switch_mode(self.kf_editor_widget.scene_mode_btn)


    def _setup_settings_monitoring(self):
        """Set up monitoring for settings changes to switch modes automatically."""
        from .settings import get_settings
        settings = get_settings(self.session)

        # Listen for settings changes
        settings.triggers.add_handler('setting changed', self._on_setting_changed)

    def _on_setting_changed(self, trigger_name, data):
        """Handle settings changes, particularly for animation mode."""
        setting_name, old_value, new_value = data
        if setting_name == 'animation_mode':
            if new_value == 'keyframe':
                self.kf_editor_widget.keyframe_mode_btn.setChecked(True)
                self.kf_editor_widget.switch_mode(self.kf_editor_widget.keyframe_mode_btn)
            else:  # scene mode
                self.kf_editor_widget.scene_mode_btn.setChecked(True)
                self.kf_editor_widget.switch_mode(self.kf_editor_widget.scene_mode_btn)


    def delete(self):
        """
        Override from super. Remove the trigger handlers before the tool is deleted.
        """
        for handler in self.handlers:
            remove_handler(handler)

        # Clean up settings trigger handler
        if hasattr(self, '_settings_handler'):
            from .settings import get_settings
            settings = get_settings(self.session)
            settings.triggers.remove_handler('setting changed', self._on_setting_changed)

        # Note: KeyframeEditorWidget cleanup is handled automatically by Qt

        super().delete()

    def take_snapshot(self, session, flags):
        return {
            'version': 1
        }

    @classmethod
    def restore_snapshot(class_obj, session, data):
        inst = class_obj(session, "Animations")
        return inst
