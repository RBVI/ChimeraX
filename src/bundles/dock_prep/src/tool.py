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

class DockPrepTool(ToolInstance):

    SESSION_SAVE = False
    #help ="help:user/tools/dockprep.html"
    help = None
    tool_name = "Dock Prep"

    def __init__(self, session, dock_prep_info):
        ToolInstance.__init__(self, session, self.tool_name)
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
        structure_layout.addWidget(QLabel("Structures to prep:"), alignment=Qt.AlignRight)
        self.structure_list = ShortASList(session)
        structure_layout.addWidget(self.structure_list, alignment=Qt.AlignLeft)
        layout.addLayout(structure_layout)
        self.tool_window.title = dock_prep_info['process name'].title()
        dp_structures = dock_prep_info['structures']
        if dp_structures is not None:
            self.structure_list.value = list(dp_structures)
            self.structure_list.setEnabled(False)

        from .settings import get_settings, defaults
        settings = get_settings(session, dock_prep_info['process name'], "base", defaults)
        layout.addWidget(QLabel("For the chosen structures, do the following:"), alignment=Qt.AlignLeft)
        self.del_solvent_button = QCheckBox("Delete solvent")
        self.del_solvent_button.setChecked(settings.del_solvent)
        layout.addWidget(self.del_solvent_button, alignment=Qt.AlignLeft)
        self.del_ions_button = QCheckBox("Delete non-complexed ions")
        self.del_ions_button.setChecked(settings.del_ions)
        layout.addWidget(self.del_ions_button, alignment=Qt.AlignLeft)
        self.standardize_button = QCheckBox('"Standardize" certain residue types:')
        self.standardize_button.setChecked(bool(settings.standardize_residues))
        layout.addWidget(self.standardize_button)
        std_text = QLabel("* selenomethionine (MSE) \N{RIGHTWARDS ARROW} methionine (MET)\n"
            "* bromo-UMP (5BU) \N{RIGHTWARDS ARROW} UMP (U)\n"
            "* methylselenyl-dUMP (UMS) \N{RIGHTWARDS ARROW} UMP (U)\n"
            "* methylselenyl-dCMP (CSL) \N{RIGHTWARDS ARROW} CMP (C)")
        std_text.setTextFormat(Qt.MarkdownText)
        layout.addWidget(std_text, alignment=Qt.AlignLeft)
        side_chain_layout = QHBoxLayout()
        side_chain_layout.setContentsMargins(0,0,0,0)
        layout.addLayout(side_chain_layout)
        self.side_chain_button = QCheckBox("Incomplete side chains:")
        sc_setting = settings.complete_side_chains
        self.side_chain_button.setChecked(bool(sc_setting))
        side_chain_layout.addWidget(self.side_chain_button, alignment=Qt.AlignLeft)
        self.rl_menu_to_arg = {}
        self.rl_arg_to_menu = {}
        self.side_chain_menu_button = QPushButton()
        sc_menu = QMenu(self.side_chain_menu_button)
        for lib_name in sorted(session.rotamers.library_names()):
            display_lib_name = session.rotamers.library(lib_name).display_name
            menu_entry = f"Replace using {display_lib_name} rotamer library"
            sc_menu.addAction(menu_entry)
            self.rl_menu_to_arg[menu_entry] = lib_name
            self.rl_arg_to_menu[lib_name] = menu_entry
        mut_ala = "Mutate residues to ALA (if CB present) or GLY"
        sc_menu.addAction(mut_ala)
        self.rl_menu_to_arg[mut_ala] = 'ala'
        self.rl_arg_to_menu['ala'] = mut_ala
        mut_gly = "Mutate residues to GLY"
        sc_menu.addAction(mut_gly)
        sc_menu.triggered.connect(lambda action, b=self.side_chain_menu_button: b.setText(action.text()))
        self.side_chain_menu_button.setMenu(sc_menu)
        self.rl_menu_to_arg[mut_gly] = 'gly'
        self.rl_arg_to_menu['gly'] = mut_gly
        if sc_setting and sc_setting is not True:
            button_text = self.rl_arg_to_menu[sc_setting]
        else:
            default_rot_lib = session.rotamers.default_command_library_name
            button_text = self.rl_arg_to_menu[default_rot_lib]
        self.side_chain_menu_button.setText(button_text)
        side_chain_layout.addWidget(self.side_chain_menu_button, alignment=Qt.AlignLeft)
        self.add_hyds_button = QCheckBox("Add hydrogens")
        self.add_hyds_button.setChecked(settings.ah)
        layout.addWidget(self.add_hyds_button, alignment=Qt.AlignLeft)
        self.add_charges_button = QCheckBox("Add charges")
        self.add_charges_button.setChecked(settings.ac)
        layout.addWidget(self.add_charges_button, alignment=Qt.AlignLeft)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        bbox = qbbox(qbbox.Ok | qbbox.Cancel | qbbox.Help)
        bbox.accepted.connect(self.dock_prep)
        bbox.rejected.connect(self.delete)
        if self.help:
            from chimerax.core.commands import run
            bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help " + self.help))
        else:
            bbox.button(qbbox.Help).setEnabled(False)
        layout.addWidget(bbox)
        self.tool_window.manage(None)

    def dock_prep(self):
        from chimerax.core.errors import UserError, CancelOperation
        self.tool_window.shown = False
        self.session.ui.processEvents()
        if not self.structures:
            if self.dock_prep_info['structures'] is None and self.structure_list.all_values:
                self.tool_window.shown = True
                raise UserError("No structures chosen for Dock Prep.")
            self.delete()
            return
        if self.side_chain_button.isChecked():
            sc_val = self.rl_menu_to_arg[self.side_chain_menu_button.text()]
        else:
            sc_val = False
        from chimerax.atomic.struct_edit import standardizable_residues
        params = {
            'del_solvent': self.del_solvent_button.isChecked(),
            'del_ions': self.del_ions_button.isChecked(),
            'standardize_residues': standardizable_residues if self.standardize_button.isChecked() else [],
            'complete_side_chains': sc_val,
            'ah': self.add_hyds_button.isChecked(),
            'ac': self.add_charges_button.isChecked(),
        }
        self.delete()
        self.dock_prep_info['callback'](self.structures, tool_settings=params)

    @property
    def structures(self):
        return self.structure_list.value
