# vim: set expandtab shiftwidth=4 softtabstop=4:

#  === UCSF ChimeraX Copyright ===
#  Copyright 2022 Regents of the University of California.
#  All rights reserved.  This software provided pursuant to a
#  license agreement containing restrictions on its disclosure,
#  duplication and use.  For details see:
#  https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
#  This notice must be embedded in or attached to all copies,
#  including partial copies, of the software or any revisions
#  or derivations thereof.
#  === UCSF ChimeraX Copyright ===
import os

from Qt.QtGui import QIcon
from Qt.QtWidgets import QVBoxLayout, QLabel, QPushButton, QLineEdit

from chimerax.core.commands import run
from chimerax.core.tools import ToolInstance
from chimerax.ui.gui import MainToolWindow

from .speech import SpeechDecoder, SpeechRecorder


class VoiceCommandTool(ToolInstance):

    SESSION_ENDURING = True
    SESSION_SAVE = False
    # help = ""

    def __init__(self, session, tool_name, **kw):
        super().__init__(session, tool_name)
        self._build_ui()

    def _build_ui(self):
        self.tool_window = MainToolWindow(self)
        parent = self.tool_window.ui_area

        layout = QVBoxLayout()
        parent.setLayout(layout)

        self.command_label = QLabel("Command:")
        self.command_text_box = QLineEdit(parent)
        self.start_recording_icon = QIcon(
            os.path.join(os.path.dirname(__file__), "resources", "red_circle.png")
        )
        self.stop_recording_icon = QIcon(
            os.path.join(os.path.dirname(__file__), "resources", "red_square.png")
        )
        self.record_button = QPushButton(
            parent, text="Record Command", icon=self.start_recording_icon
        )
        self.confirm_button = QPushButton("Confirm Command")

        layout.addWidget(self.command_label)
        layout.addWidget(self.command_text_box)
        layout.addWidget(self.record_button)
        layout.addWidget(self.confirm_button)

        self.record_button.clicked.connect(self.record_command)
        self.confirm_button.clicked.connect(self.execute_command)

        self.tool_window.manage()

    def record_command(self):
        self.record_button.setText("Stop Recording")
        self.record_button.setIcon(self.stop_recording_icon)
        self.record_button.clicked.disconnect(self.record_command)
        self.record_button.clicked.connect(self.stop_recording)
        self.recorder = SpeechRecorder()
        self.recorder.record()

    def stop_recording(self):
        self.record_button.setText("Record Command")
        self.record_button.setIcon(self.start_recording_icon)
        self.record_button.clicked.disconnect(self.stop_recording)
        self.record_button.clicked.connect(self.record_command)
        self.recorder.close()
        decoder = SpeechDecoder()
        result = decoder.decode_frames(self.recorder.get_frames())
        self.command_text_box.setText(result.getText())

    def execute_command(self):
        command = self.command_text_box.text()
        run(self.session, command)
        self.command_text_box.setText("")

    def insert_command(self, command):
        self.command_text_box.setText(command)
