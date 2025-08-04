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
        #self.animation_mgr = self.session.get_state_manager("animations")

        from chimerax.ui import MainToolWindow
        self.tool_window = MainToolWindow(self)

        # test scene thumbnails
        self.build_ui()

        # Register handlers for the triggers
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

        # Keyframe editor graphics view widget.
        self.kf_editor_widget = KeyframeEditorWidget(self.session)
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

    def delete(self):
        """
        Override from super. Remove the trigger handlers before the tool is deleted.
        """
        for handler in self.handlers:
            remove_handler(handler)
        if self.kf_editor_widget is not None:
            self.kf_editor_widget.remove_handlers()
        super().delete()

    def take_snapshot(self, session, flags):
        return {
            'version': 1
        }

    @classmethod
    def restore_snapshot(class_obj, session, data):
        inst = class_obj(session, "Animations")
        return inst
