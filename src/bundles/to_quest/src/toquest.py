# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
# User interface for sending scenes to Quest VR headset for viewing with
# Lookie app
#
from chimerax.core.tools import ToolInstance

# ------------------------------------------------------------------------------
#
class ToQuest(ToolInstance):
    SESSION_ENDURING = True

    # help = 'help:user/tools/markerplacement.html'

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self, close_destroys=False)
        self.tool_window = tw
        parent = tw.ui_area
        
        from Qt.QtWidgets import QFrame, QHBoxLayout, QPushButton

        layout = QHBoxLayout(parent)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        parent.setLayout(layout)

        layout.addStretch(1)

        # Send to Quest button
        sb = QPushButton('Send to Quest', parent)
        sb.clicked.connect(self._send_to_quest)
        layout.addWidget(sb)

        layout.addStretch(1)
        
        tw.manage(placement="side")

    @classmethod
    def get_singleton(self, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, ToQuest, 'Send to Quest', create=create)

    def _send_to_quest(self):
        # Save current scene.
        from os.path import expanduser, sep
        path = expanduser('~/Desktop/scene.glb').replace(sep, '/')
        from chimerax.core.commands import run
        run(self.session, f'save {path}')

        #
        # To setup adb for wireless connection.
        # 1) First plug in quest to computer with usb.
        # 2) Click Allow access in Quest headset.
        # 3) Check Quest IP address in headset clicking the wifi symbol in the small gui pane,
        #    then clicking the connected wifi, then scroll down to click Advanced.
        # 4) Use adb command "adb tcpip 5555"
        # 5) Use adb command "adb 169.230.21.238" to specify the quest IP address.
        # 6) Check connected devices with "adb devices"
        # 7) Disconnect usb cable from quest.
        #
        adb = 'C:/Program Files/Unity/Hub/Editor/2022.2.5f1/Editor/Data/PlaybackEngines/AndroidPlayer/SDK/platform-tools/adb.exe'
#        lookie_dir = '/sdcard/Android/data/com.UCSF.Lookie/files'
        lookie_dir = '/sdcard/Android/data/com.UCSF.LookieAR/files'
        cmd = f'"{adb}" push {path} {lookie_dir}'
        self.session.logger.info(f'Running command: {cmd}')

        # all output is on stderr, but Windows needs all standard I/O to
        # be redirected if one is, so stdout is a pipe too
        args = [adb, "push", path, lookie_dir]
        from subprocess import Popen, PIPE, DEVNULL
        p = Popen(args, stdin=DEVNULL, stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()
        exit_code = p.returncode

        output = 'stdout:\n%s\nstderr:\n%s' % (out.decode('utf-8'), err.decode('utf-8'))
        if exit_code or True:
            self.session.logger.info(output)

#        import os
#        os.system(cmd)
        
def to_quest_panel(session, create = True):
  return ToQuest.get_singleton(session, create)
