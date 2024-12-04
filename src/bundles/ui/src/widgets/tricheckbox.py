# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from Qt.QtWidgets import QCheckBox
from Qt.QtCore import Qt

class TwoThreeStateCheckBox(QCheckBox):
    """QCheckBox that can show a partially checked state, but does include the partially checked state
       when the user clicks on it"""
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.setTristate(True)

    def nextCheckState(self):
        # it seems that just returning the next check state is insuffient,
        # you also have to explicitly set it
        if self.checkState() == Qt.Checked:
            self.setCheckState(Qt.Unchecked)
        else:
            self.setCheckState(Qt.Checked)
