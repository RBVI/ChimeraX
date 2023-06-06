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
    help ="help:user/tools/dockprep.html"
    tool_name = "Dock Prep"

    def __init__(self, session, *, dock_prep_info=None):
        ToolInstance.__init__(self, session, self.tool_name)
        if dock_prep_info is None:
            def callback(structures, tool_settings, *, session=session):
                from . import dock_prep_caller
                dock_prep_caller(session, structures, _from_tool=True, **tool_settings)
            dock_prep_info = {
                'structures': None,
                'process name': 'dock prep',
                'callback': callback
            }
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
        self.del_alt_locs_button = QCheckBox("Delete non-current alternate locations")
        self.del_alt_locs_button.setChecked(settings.del_alt_locs)
        layout.addWidget(self.del_alt_locs_button, alignment=Qt.AlignLeft)
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
        self.citation_widgets = { "showing": None }
        self.side_chain_button = QCheckBox("Incomplete side chains:")
        sc_setting = settings.complete_side_chains
        self.side_chain_button.setChecked(bool(sc_setting))
        side_chain_layout.addWidget(self.side_chain_button, alignment=Qt.AlignLeft)
        self.sc_menu_to_arg = {}
        self.sc_arg_to_menu = {}
        self.side_chain_menu_button = QPushButton()
        sc_menu = QMenu(self.side_chain_menu_button)
        sc_menu.triggered.connect(lambda action, f=self._lib_change_cb: f(action.text()))
        sc_menu.aboutToShow.connect(self._fill_sidechain_menu)
        self.side_chain_menu_button.setMenu(sc_menu)
        self._fill_sidechain_menu()
        if sc_setting and sc_setting is not True:
            button_text = self.sc_arg_to_menu[sc_setting]
        else:
            default_rot_lib = session.rotamers.default_command_library_name
            button_text = self.sc_arg_to_menu[default_rot_lib]
        side_chain_layout.addWidget(self.side_chain_menu_button, alignment=Qt.AlignLeft)
        self.add_hyds_button = QCheckBox("Add hydrogens")
        self.add_hyds_button.setChecked(settings.ah)
        layout.addWidget(self.add_hyds_button, alignment=Qt.AlignLeft)
        self.add_charges_button = QCheckBox("Add charges")
        self.add_charges_button.setChecked(settings.ac)
        layout.addWidget(self.add_charges_button, alignment=Qt.AlignLeft)
        self.write_mol2_button = QCheckBox("Write Mol2 file")
        plain_dock_prep = dock_prep_info['process name'] == "dock prep"
        self.write_mol2_button.setChecked(plain_dock_prep and settings.write_mol2)
        layout.addWidget(self.write_mol2_button, alignment=Qt.AlignLeft)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        self.bbox = bbox = qbbox(qbbox.Ok | qbbox.Cancel | qbbox.Help)
        bbox.accepted.connect(self.dock_prep)
        bbox.rejected.connect(self.delete)
        if self.help:
            from chimerax.core.commands import run
            bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help " + self.help))
        else:
            bbox.button(qbbox.Help).setEnabled(False)
        layout.addWidget(bbox)
        self.tool_window.manage(None)
        # self.bbox has to be defined before we call this, _and_ the window has to be floating, otherwise
        # shrink_to_fit() shrinks the other docked widgets
        self._lib_change_cb(button_text)

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
            sc_val = self.sc_menu_to_arg[self.side_chain_menu_button.text()]
        else:
            sc_val = False
        from chimerax.atomic.struct_edit import standardizable_residues
        params = {
            'del_solvent': self.del_solvent_button.isChecked(),
            'del_ions': self.del_ions_button.isChecked(),
            'del_alt_locs': self.del_alt_locs_button.isChecked(),
            'standardize_residues': standardizable_residues if self.standardize_button.isChecked() else [],
            'complete_side_chains': sc_val,
            'ah': self.add_hyds_button.isChecked(),
            'ac': self.add_charges_button.isChecked(),
            'write_mol2': self.write_mol2_button.isChecked(),
        }
        if self.dock_prep_info['process name'] == "dock prep":
            from .settings import get_settings, defaults
            settings = get_settings(self.session, self.dock_prep_info['process name'], "base", defaults)
            settings.write_mol2 = params['write_mol2']
        try:
            self.dock_prep_info['callback'](self.structures, tool_settings=params)
        finally:
            self.delete()

    @property
    def structures(self):
        return self.structure_list.value

    def _fill_sidechain_menu(self):
        sc_menu = self.side_chain_menu_button.menu()
        sc_menu.clear()
        mgr = self.session.rotamers
        for lib_name in sorted(mgr.library_names()):
            menu_entry = "Replace using %s rotamer library" % mgr.ui_name(lib_name)
            sc_menu.addAction(menu_entry)
            self.sc_menu_to_arg[menu_entry] = lib_name
            self.sc_arg_to_menu[lib_name] = menu_entry
        mut_ala = "Mutate residues to ALA (if CB present) or GLY"
        sc_menu.addAction(mut_ala)
        self.sc_menu_to_arg[mut_ala] = 'ala'
        self.sc_arg_to_menu['ala'] = mut_ala
        mut_gly = "Mutate residues to GLY"
        sc_menu.addAction(mut_gly)
        self.sc_menu_to_arg[mut_gly] = 'gly'
        self.sc_arg_to_menu['gly'] = mut_gly

    def _lib_change_cb(self, lib_text):
        self.side_chain_menu_button.setText(lib_text)
        layout = self.tool_window.ui_area.layout()
        prev_cite = self.citation_widgets["showing"]
        if prev_cite:
            layout.removeWidget(prev_cite)
            prev_cite.hide()
        arg_val = self.sc_menu_to_arg[lib_text]
        try:
            new_cite = self.citation_widgets[arg_val]
        except KeyError:
            from chimerax.rotamers import NoRotamerLibraryError
            try:
                lib = self.session.rotamers.library(arg_val)
            except NoRotamerLibraryError:
                new_cite = None
            else:
                new_cite = self._make_citation_widget(lib)
            self.citation_widgets[arg_val] = new_cite
        if new_cite:
            from Qt.QtCore import Qt
            layout.insertWidget(layout.indexOf(self.bbox), new_cite, alignment=Qt.AlignCenter)
            new_cite.show()
        self.citation_widgets["showing"] = new_cite
        self.tool_window.shrink_to_fit()

    def _make_citation_widget(self, lib):
        if not lib.citation:
            return None
        from chimerax.ui.widgets import Citation
        return Citation(self.session, lib.citation, prefix="Publications using %s rotamers should cite:"
            % self.session.rotamers.ui_name(lib.name), pubmed_id=lib.cite_pubmed_id)
