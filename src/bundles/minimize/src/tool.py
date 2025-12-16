# vim: set expandtab shiftwidth=4 softtabstop=4:

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

from chimerax.core.tools import ToolInstance

class MinimizeTool(ToolInstance):

    help = "help:user/tools/minimizestructure.html"

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)
        self.settings = MinimizeSettings(session, "Minimize tool")

        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area

        from Qt.QtWidgets import QVBoxLayout, QCheckBox, QHBoxLayout, QSpinBox, QLabel
        from Qt.QtCore import Qt
        layout = QVBoxLayout()
        parent.setLayout(layout)
        from chimerax.atomic.widgets import AtomicStructureMenuButton as ASMB
        self.structure_button = ASMB(session)
        layout.addWidget(self.structure_button)

        steps_layout = QHBoxLayout()
        steps_layout.setSpacing(0)
        steps_layout.setContentsMargins(0,0,0,0)
        do_convergence = self.settings.steps == 0
        self.convergence_button = QCheckBox("Minimize until convergence")
        self.convergence_button.setChecked(do_convergence)
        self.convergence_button.stateChanged.connect(self._convergence_changed)
        steps_layout.addWidget(self.convergence_button)
        steps_layout.addSpacing(20)
        self.max_steps_label = QLabel("Maximum # steps: ")
        steps_layout.addWidget(self.max_steps_label, alignment=Qt.AlignRight)
        self.max_steps_box = QSpinBox()
        self.max_steps_box.setRange(1, 999999)
        self.max_steps_box.setValue(1000 if self.settings.steps == 0 else self.settings.steps)
        self.max_steps_label.setEnabled(not do_convergence)
        self.max_steps_box.setEnabled(not do_convergence)
        steps_layout.addWidget(self.max_steps_box, alignment=Qt.AlignLeft)
        layout.addLayout(steps_layout)

        self.update_structure_button = QCheckBox("Update structure during minimization")
        self.update_structure_button.setChecked(self.settings.update_structure)
        layout.addWidget(self.update_structure_button)

        self.log_energies_button = QCheckBox("Report energies to Log")
        self.log_energies_button.setChecked(self.settings.log_energies)
        layout.addWidget(self.log_energies_button)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        bbox = qbbox(qbbox.Ok | qbbox.Cancel | qbbox.Help)
        bbox.accepted.connect(self.minimize)
        bbox.rejected.connect(self.delete)
        if self.help:
            from chimerax.core.commands import run
            bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help " + self.help))
        else:
            bbox.button(qbbox.Help).setEnabled(False)
        layout.addWidget(bbox)

        tw.manage(placement=None)

    def minimize(self):
        try:
            self.tool_window.shown = False
            self.session.ui.processEvents()
            struct = self.structure_button.value
            if not struct:
                from chimerax.core.errors import UserError
                raise UserError("No structure to minimize")
            cmd = f"minimize {struct.atomspec}"
            if self.convergence_button.isChecked():
                self.settings.steps = 0
            else:
                self.settings.steps = self.max_steps_box.value()
                cmd += f" maxSteps {self.settings.steps}"
            if self.update_structure_button.isChecked():
                self.settings.update_structure = True
            else:
                self.settings.update_structure = False
                cmd += " liveUpdates false"
            if self.log_energies_button.isChecked():
                self.settings.log_energies = True
                cmd += " logEnergy true"
            else:
                self.settings.log_energies = False
            from chimerax.core.commands import run
            run(self.session, cmd)
        finally:
            self.delete()

    def _convergence_changed(self):
        do_convergence = self.convergence_button.isChecked()
        self.max_steps_label.setEnabled(not do_convergence)
        self.max_steps_box.setEnabled(not do_convergence)


from chimerax.core.settings import Settings
class MinimizeSettings(Settings):
    AUTO_SAVE = {
        'log_energies': False,
        'steps': 0,
        'update_structure': True
    }
