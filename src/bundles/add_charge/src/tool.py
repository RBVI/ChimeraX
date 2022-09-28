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

class AddChargeTool(ToolInstance):

    SESSION_SAVE = False
    #help ="help:user/tools/addhydrogens.html"

    def __init__(self, session, tool_name, *, dock_prep_info=None):
        ToolInstance.__init__(self, session, tool_name)
        self.dock_prep_info = dock_prep_info

        from chimerax.ui import MainToolWindow
        self.tool_window = MainToolWindow(self)
        parent = self.tool_window.ui_area

        from Qt.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QGroupBox, QButtonGroup
        from Qt.QtWidgets import QRadioButton, QPushButton, QMenu, QWidget
        from Qt.QtCore import Qt

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        parent.setLayout(layout)

        from chimerax.atomic.widgets import AtomicStructureListWidget
        class ShortASList(AtomicStructureListWidget):
            def sizeHint(self):
                hint = super().sizeHint()
                hint.setHeight(hint.height()//2)
                return hint
        structure_layout = QHBoxLayout()
        structure_layout.addWidget(QLabel("Assign charges to:"), alignment=Qt.AlignRight)
        self.structure_list = ShortASList(session)
        structure_layout.addWidget(self.structure_list, alignment=Qt.AlignLeft)
        layout.addLayout(structure_layout)
        if dock_prep_info is not None:
            self.tool_window.title = "%s for %s" % (tool_name, dock_prep_info.process_name.capitalize())
            self.structure_list.setEnabled(False)

        self.standardize_button = QCheckBox('"Standardize" certain residues:')
        self.standardize_button.setChecked(True)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        bbox = qbbox(qbbox.Ok | qbbox.Cancel | qbbox.Help)
        bbox.accepted.connect(self.add_charges)
        bbox.rejected.connect(self.delete)
        from chimerax.core.commands import run
        bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help " + self.help))
        bbox.button(qbbox.Help).setEnabled(False)
        layout.addWidget(bbox)
        self.tool_window.manage(None)

    def add_charges(self):
        from chimerax.core.errors import UserError
        self.tool_window.shown = False
        self.session.ui.processEvents()
        if not self.structures:
            if self.dock_prep_info is None:
                self.tool_window.shown = True
                raise UserError("No structures chosen for hydrogen addition.")
            self.delete()
            return
        from chimerax.core.commands import run, concise_model_spec
        cmd = "addh %s" % concise_model_spec(self.session, self.structures)
        if not self.isolation.isChecked():
            cmd += " inIsolation false"
        if self.method_group.checkedButton() == self.steric_method:
            cmd += " hbond false"
        for res_name, widgets in self.prot_widget_lookup.items():
            box, grp = widgets
            if not grp.checkedButton().text().startswith("Residue-name-based"):
                cmd += " use%sName false" % self.prot_arg_lookup[res_name].capitalize()
        from .cmd import metal_dist_default
        if self.metal_option.value != metal_dist_default:
            cmd += " metalDist %g" % self.metal_option.value
        if self.template_checkbox.isChecked():
            cmd += " template true"
        run(self.session, cmd)
        self.delete()
        if self.dock_prep_info is not None:
            #TODO: continue dock prep call chain
            pass

    def delete(self):
        ToolInstance.delete(self)

    @property
    def structures(self):
        if self.dock_prep_info is None:
            return self.structure_list.value
        return self.dock_prep_info['structures']

    def _protonation_res_change(self, res_name):
        self.protonation_res_button.setText(res_name)
        for box, grp in self.prot_widget_lookup.values():
            box.setHidden(True)
        box, grp = self.prot_widget_lookup[res_name]
        box.setHidden(False)

    def _toggle_options(self, *args, **kw):
        self.options_area.setHidden(not self.options_area.isHidden())
