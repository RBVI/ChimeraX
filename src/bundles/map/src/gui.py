# vim: set expandtab shiftwidth=4 softtabstop=4:

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

from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel
class SaveOptionsWidget(QFrame):
    def __init__(self, session):
        super().__init__()

        mlayout = QHBoxLayout()
        mlayout.setContentsMargins(0,0,0,0)
        mlayout.setSpacing(10)
        self.setLayout(mlayout)

        sm = QLabel('Map')
        mlayout.addWidget(sm)
        from chimerax.ui.widgets import ModelMenuButton
        from chimerax.map import Volume
        self._map_menu = mm = ModelMenuButton(session, class_filter = Volume)
        mlayout.addWidget(mm)
        mlayout.addStretch(1)    # Extra space at end

    def options_string(self):
        map = self._map_menu.value
        if map is None:
            from chimerax.core.errors import UserError
            raise UserError("No map to save")
        return 'model #%s' % map.id_string
