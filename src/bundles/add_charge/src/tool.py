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
    help = None

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
        layout.setSpacing(5)
        parent.setLayout(layout)

        from chimerax.atomic.widgets import AtomicStructureListWidget
        class ShortASList(AtomicStructureListWidget):
            def sizeHint(self):
                hint = super().sizeHint()
                hint.setHeight(hint.height()//2)
                return hint
        structure_layout = QHBoxLayout()
        structure_layout.addWidget(QLabel("Assign charges to:"), alignment=Qt.AlignRight)
        list_layout = QVBoxLayout()
        structure_layout.addLayout(list_layout)
        self.structure_list = ShortASList(session)
        list_layout.addWidget(self.structure_list, alignment=Qt.AlignLeft)
        self.sel_restrict = QCheckBox("Also restrict to selection")
        self.sel_restrict.setChecked(dock_prep_info is None)
        list_layout.addWidget(self.sel_restrict, alignment=Qt.AlignCenter)
        layout.addLayout(structure_layout)
        if dock_prep_info is not None:
            self.tool_window.title = "%s for %s" % (tool_name, dock_prep_info['process_name'].capitalize())
            self.structure_list.setEnabled(False)
            self.sel_restrict.setHidden(True)
        else:
            # Dock Prep handles standardization directly
            self.standardize_button = QCheckBox('"Standardize" certain residue types:')
            self.standardize_button.setChecked(True)
            layout.addWidget(self.standardize_button)
            std_text = QLabel("* selenomethionine (MSE) \N{RIGHTWARDS ARROW} methionine (MET)\n"
                "* bromo-UMP (5BU) \N{RIGHTWARDS ARROW} UMP (U)\n"
                "* methylselenyl-dUMP (UMS) \N{RIGHTWARDS ARROW} UMP (U)\n"
                "* methylselenyl-dCMP (CSL) \N{RIGHTWARDS ARROW} CMP (C)")
            std_text.setTextFormat(Qt.MarkdownText)
            layout.addWidget(std_text)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        bbox = qbbox(qbbox.Ok | qbbox.Cancel | qbbox.Help)
        bbox.accepted.connect(self.add_charges)
        bbox.rejected.connect(self.delete)
        if self.help:
            from chimerax.core.commands import run
            bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help " + self.help))
        else:
            bbox.button(qbbox.Help).setEnabled(False)
        layout.addWidget(bbox)
        self.tool_window.manage(None)

    def add_charges(self):
        from chimerax.core.errors import UserError, CancelOperation
        self.tool_window.shown = False
        self.session.ui.processEvents()
        if not self.structures:
            if self.dock_prep_info is None and self.structure_list.all_values:
                self.tool_window.shown = True
                raise UserError("No structures chosen for charge addition.")
            self.delete()
            return
        sel_restrict = self.sel_restrict.isChecked()
        from chimerax.atomic import AtomicStructures, selected_residues
        if sel_restrict:
            selected = selected_residues(self.session)
            if not selected:
                sel_restrict = False
        residues = AtomicStructures(self.structures).residues
        if sel_restrict:
            residues = residues.intersect(selected)
            if not residues:
                self.tool_window.shown = True
                raise UserError("None of the selected residues are in the structures being assigned charges")
        if self.dock_prep_info is None:
            process_info = {
                'process name': "adding charges",
                'structures': self.structures,
                'callback': lambda f=self._finish_add_charge, r=residues: f(r)
            }
            from chimerax.addh.tool import check_no_hyds
            try:
                check_no_hyds(self.session, residues, process_info, help=self.help)
            except CancelOperation:
                self.delete()
        else:
            # Dock prep has add hydrogens built in, so don't check
            self._finish_add_charge(residues)

    @property
    def structures(self):
        if self.dock_prep_info is None:
            return self.structure_list.value
        return self.dock_prep_info['structures']

    def _finish_add_charge(self, residues):
        if not residues:
            self.delete()
            return
        from chimerax.atomic.struct_edit import standardizable_residues
        standardize = self.standardize_button.isChecked() if self.dock_prep_info is None else False
        params = {
            'standardize_residues': standardizable_residues if standardize else [],
        }
        sel_restrict = self.sel_restrict.isChecked()
        from chimerax.core.commands import concise_model_spec
        self.session.logger.info("Closest equivalent command: <b>addcharge %s%s standardizeResidues %s</b>"
            % (concise_model_spec(self.session, self.structures), " & sel" if sel_restrict else "",
            ",".join(standardizable_residues) if standardize else "none"), is_html=True)
        from .charge import add_standard_charges
        non_std = add_standard_charges(self.session, residues=residues, **params)
        if non_std:
            #TODO: launch non-standard tool
            raise NotImplementedError("Non-standard charges tool")
        self.delete()
        if (not non_std) and self.dock_prep_info is not None:
            self.dock_prep_info['callback'](tool_settings=params)
