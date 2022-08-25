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

from chimerax.core.tools import ToolInstance

class AddHTool(ToolInstance):

    SESSION_SAVE = False

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name, *, dock_prep_info=None)
        self.dock_prep_info = dock_prep_info

        from chimerax.ui import MainToolWindow
        self.tool_window = MainToolWindow(self)
        parent = self.tool_window.ui_area

        from Qt.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QGroupBox, QButtonGroup
        from Qt.QtWidgets import QRadioButton
        from Qt.QtCore import Qt

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        parent.setLayout(layout)

        if dock_prep_info is None:
            from chimerax.atomic.widgets import AtomicStructureListWidget
            class ShortASList(AtomicStructureListWidget):
                def sizeHint(self):
                    hint = super().sizeHint()
                    hint.setHeight(hint.height()//2)
                    return hint
            structure_layout = QHBoxLayout()
            structure_layout.addWidget(QLabel("Add hydrogens to:"), alignment=Qt.AlignRight)
            self.structure_list = ShortASList(session)
            structure_layout.addWidget(self.structure_list, alignment=Qt.AlignLeft)
            layout.addLayout(structure_layout)
        self.isolation = QCheckBox("Consider each model in isolation from all others")
        self.isolation.setChecked(True)
        layout.addWidget(self.isolation, alignment=Qt.AlignCenter)

        layout.addSpacing(10)

        method_groupbox = QGroupBox("Method")
        method_layout = QVBoxLayout()
        method_groupbox.setLayout(method_layout)
        layout.addWidget(method_groupbox, alignment=Qt.AlignCenter)
        self.method_group = QButtonGroup()
        self.steric_method = QRadioButton("steric only")
        self.method_group.addButton(self.steric_method)
        method_layout.addWidget(self.steric_method, alignment=Qt.AlignLeft)
        self.hbond_method = QRadioButton("also consider H-bonds (slower)")
        self.method_group.addButton(self.hbond_method)
        method_layout.addWidget(self.hbond_method, alignment=Qt.AlignLeft)
        self.hbond_method.setChecked(True)

        layout.addSpacing(10)

        #TODO: protonation states

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        bbox = qbbox(qbbox.Ok | qbbox.Cancel | qbbox.Help)
        # connected function will call self.delete() if no errors
        #bbox.accepted.connect(self.launch_modeller)
        bbox.rejected.connect(self.delete)
        from chimerax.core.commands import run
        bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help " + self.help))
        bbox.button(qbbox.Help).setEnabled(False)
        layout.addWidget(bbox)
        self.tool_window.manage(None)

    def delete(self):
        ToolInstance.delete(self)

    @property
    def structures(self):
        if self.dock_prep_info is None:
            return self.structure_list.value
        return self.dock_prep_info['structures']
