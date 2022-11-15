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
            dp_structures = dock_prep_info['structures']
            if dp_structures is not None:
                self.structure_list.value = list(dp_structures)
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
        return self.structure_list.value

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
            % (concise_model_spec(self.session, self.structures,
            relevant_types=self.structures[0].__class__), " & sel" if sel_restrict else "",
            ",".join(standardizable_residues) if standardize else "none"), is_html=True)
        from .charge import add_standard_charges
        non_std = add_standard_charges(self.session, residues=residues, **params)
        if non_std:
            if self.dock_prep_info is not None:
                from chimerax.atomic import AtomicStructures
                self.dock_prep_info['structures'] = AtomicStructures(self.structures)
            AddNonstandardChargesTool(self.session, "Add Non-Standard Charges", non_std,
                dock_prep_info=self.dock_prep_info, main_params=params)
        self.delete()
        if (not non_std) and self.dock_prep_info is not None:
            from chimerax.atomic import AtomicStructures
            self.dock_prep_info['callback'](AtomicStructures(self.structures), tool_settings=params)

class AddNonstandardChargesTool(ToolInstance):

    SESSION_SAVE = False
    #help ="help:user/tools/addhydrogens.html"
    help = None

    def __init__(self, session, tool_name, non_std_info, *, dock_prep_info=None, main_params=None):
        ToolInstance.__init__(self, session, tool_name)
        self.dock_prep_info = dock_prep_info
        self.main_params = main_params
        self.non_std_info = non_std_info

        from chimerax.ui import MainToolWindow
        self.tool_window = MainToolWindow(self)
        parent = self.tool_window.ui_area

        from Qt.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QGroupBox, QButtonGroup
        from Qt.QtWidgets import QRadioButton, QPushButton, QMenu, QWidget, QFrame, QGridLayout
        from Qt.QtCore import Qt

        layout = QVBoxLayout()
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(5)
        parent.setLayout(layout)

        if dock_prep_info is not None:
            self.tool_window.title = "%s for %s" % (tool_name, dock_prep_info['process_name'].capitalize())

        charge_frame = QFrame()
        charge_frame.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Plain)
        layout.addWidget(charge_frame, alignment=Qt.AlignCenter)
        frame_layout = QGridLayout()
        frame_layout.setSpacing(0)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        charge_frame.setLayout(frame_layout)
        for col, title in enumerate(["Residue", "Net Charge"]):
            l = QLabel(title, alignment=Qt.AlignCenter)
            l.setFrameStyle(QFrame.Shape.Panel|QFrame.Shadow.Raised)
            l.setLineWidth(2)
            frame_layout.addWidget(l, 0, col)
        from . import estimate_net_charge
        self.charge_widgets = {}
        for text, residues in non_std_info.items():
            enc = estimate_net_charge(residues[0].atoms)
            row = frame_layout.rowCount()
            frame_layout.addWidget(QLabel(text), row, 0, alignment=Qt.AlignCenter)
            button = self.charge_widgets[text] = QPushButton("%+d" % enc if enc else "+0")
            frame_layout.addWidget(button, row, 1, alignment=Qt.AlignCenter)
            charge_menu = QMenu(button)
            button.setMenu(charge_menu)
            charge_menu.triggered.connect(lambda action, button=button: button.setText(action.text()))
            low_charge = min(-9, enc-5)
            high_charge = max(9, enc+5)
            for c in range(low_charge, high_charge+1):
                charge_menu.addAction("%+d" % c if c else "0")

        instructions = QLabel("Please specify the net charges for the above residues"
            " so that their atomic partial charges can be computed.")
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Since QFormLayout doesn't center label with widget, cheat ...
        method_group = QGroupBox("Charge method")
        method_layout = QHBoxLayout()
        method_layout.setContentsMargins(0,0,0,0)
        method_group.setLayout(method_layout)
        method_group.setAlignment(Qt.AlignHCenter)
        from chimerax.ui.options import SettingsPanel, SymbolicEnumOption
        from .cmd import ChargeMethodArg
        from .settings import get_settings
        panel = SettingsPanel(scrolled=False, contents_margins=(0,0,0,0), buttons=False)
        settings =  get_settings(session)
        class MethodOption(SymbolicEnumOption):
            labels = [x.upper() if '-' in x else x.capitalize() for x in ChargeMethodArg.values]
            values = ChargeMethodArg.values
        self.method_option = MethodOption("", settings.method, None, as_radio_buttons=True,
            horizontal_radio_buttons=True, attr_name="method", settings=settings)
        panel.add_option(self.method_option)
        method_layout.addWidget(panel)
        layout.addWidget(method_group)

        from chimerax.ui.widgets import Citation
        layout.addWidget(Citation(session, "Wang, J., Wang, W., Kollman, P.A., and Case, D.A. (2006)\n"
            "Automatic atom type and bond type perception in molecular mechanical calculations\n"
            "Journal of Molecular Graphics and Modelling, 25, 247-260.",
            prefix="Charges are computed using ANTECHAMBER.\n"
            "Publications using ANTECHAMBER charges should cite:", pubmed_id=16458552))

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
        from .charge import add_nonstandard_res_charges
        self.tool_window.shown = False
        self.session.ui.processEvents()
        method = self.method_option.value
        for text, residues in self.non_std_info.items():
            residues = [r for r in residues if not r.deleted]
            if residues:
                charge = int(self.charge_widgets[text].text())
                add_nonstandard_res_charges(self.session, residues, charge, method=method)
        self.delete()
        if self.dock_prep_info is not None:
            self.main_params['method'] = method
            self.dock_prep_info['callback'](self.dock_prep_info['structures'],
                tool_settings=self.main_params)
