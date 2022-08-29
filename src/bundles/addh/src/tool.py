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

        # In Chimera, there was an option for specifying protonation states on a residue-by-residue
        # basis in the GUI.  Not implementing that here (and in command) until a proven need exists.
        self.options_area = QWidget()
        layout.addWidget(self.options_area)
        self.options_area.setHidden(True)
        options_layout = QVBoxLayout()
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.setSpacing(0)
        self.options_area.setLayout(options_layout)
        protonation_res_layout = QHBoxLayout()
        options_layout.addLayout(protonation_res_layout)
        protonation_res_layout.addWidget(QLabel("Protonation states for:"), alignment=Qt.AlignRight)
        self.protonation_res_button = QPushButton()
        protonation_res_layout.addWidget(self.protonation_res_button, alignment=Qt.AlignLeft)
        prot_menu = QMenu(self.protonation_res_button)
        self.protonation_res_button.setMenu(prot_menu)
        prot_menu.triggered.connect(lambda act: self._protonation_res_change(act.text()))
        self.prot_widget_lookup = {}
        self.prot_arg_lookup = {}
        for res_abbr, res_name, explanation, charged in [
            ("ASP", "aspartic acid", "ASP/ASH = negatively charged/neutral [protonated OD2]", True),
            ("CYS", "cysteine", "CYS/CYM = unspecified/negatively charged", False),
            ("GLU", "glutamic acid", "GLU/GLH = negatively charged/neutral [protonated OE2]", True),
            ("HIS", "histidine", "HIS/HID/HIE/HIP = unspecified/delta/epsilon/both", False),
            ("LYS", "lysine", "LYS/LYN = positively charged/neutral", True),
        ]:
            self.prot_arg_lookup[res_name] = res_abbr
            prot_menu.addAction(res_name)
            box = QGroupBox()
            options_layout.addWidget(box)
            box.setHidden(True)
            box_layout = QVBoxLayout()
            box.setLayout(box_layout)
            grp = QButtonGroup()
            self.prot_widget_lookup[res_name] = (box, grp)
            b1 = QRadioButton(f"Residue-name-based\n({explanation})")
            box_layout.addWidget(b1, alignment=Qt.AlignLeft)
            grp.addButton(b1)
            b2 = QRadioButton("Charged" if charged else "Unspecified (determined by method)")
            box_layout.addWidget(b2, alignment=Qt.AlignLeft)
            grp.addButton(b2)
            b1.setChecked(True)
        self._protonation_res_change("histidine")

        from chimerax.ui.options import OptionsPanel, FloatOption, BooleanOption
        self.options_panel = OptionsPanel(sorting=False, scrolled=False)
        self.options_panel.layout().setSpacing(20)
        from .cmd import metal_dist_default
        self.metal_option = FloatOption(
            "Strip proton from heavy atom X near metal M if X\N{LEFT RIGHT ARROW}M\n"
            "distance less than entry value, and...\n"
            "    \N{BULLET} X is electronegative\n"
            "    \N{BULLET} the X-H-M angle would be >120°\n",
            metal_dist_default, None, min=0.0, max=99.9)
        self.options_panel.add_option(self.metal_option)
        options_layout.addWidget(self.options_panel)
        self.template_option = BooleanOption("Base non-standard residue atom typing on\n"
            "idealized coordinates rather than actual coordinates", False, None,
            balloon="If a non-standard residue has an entry in the PDB Chemical Component Dictionary,\n"
            "use the idealized coordinates from the entry for atom typing rather than the actual\n"
            "coordinates from the structure.  This means the residue name has to correspond to the\n"
            "component name in the dictionary.")
        self.options_panel.add_option(self.template_option)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        bbox = qbbox(qbbox.Ok | qbbox.Cancel | qbbox.Help)
        options_button = bbox.addButton("Options", qbbox.ActionRole)
        options_button.clicked.connect(self._toggle_options)
        bbox.accepted.connect(self.add_h)
        bbox.rejected.connect(self.delete)
        from chimerax.core.commands import run
        bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help " + self.help))
        bbox.button(qbbox.Help).setEnabled(False)
        layout.addWidget(bbox)
        self.tool_window.manage(None)

    def add_h(self):
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
        if self.template_option.value:
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
