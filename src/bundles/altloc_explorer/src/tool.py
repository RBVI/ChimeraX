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
from chimerax.core.settings import Settings
from Qt.QtWidgets import QVBoxLayout, QGridLayout, QHBoxLayout, QLabel, QButtonGroup, QRadioButton, QWidget
from Qt.QtWidgets import QPushButton, QScrollArea
from Qt.QtCore import Qt
from chimerax.core.commands import run

class AltlocExplorerSettings(Settings):
    AUTO_SAVE = {
        "show_hbonds": True,
    }
class AltlocExplorerTool(ToolInstance):

    help = "help:user/tools/altlocexplorer.html"

    hbonds_group_name = "altloc H-bonds"

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area

        main_layout = QVBoxLayout()
        main_layout.setSpacing(2)
        parent.setLayout(main_layout)

        from chimerax.atomic.widgets import AtomicStructureMenuButton as ASMB
        self._structure_button = button = ASMB(session)
        button.value_changed.connect(self._structure_change)
        main_layout.addWidget(button)
        widgets_layout = QHBoxLayout()
        main_layout.addLayout(widgets_layout)
        # the altlocs get their own layout so they can be replaced without moving to the end
        self._altlocs_layout = QHBoxLayout()
        widgets_layout.addLayout(self._altlocs_layout, stretch=1)
        self._no_structure_label = QLabel("No atomic model chosen")
        self._altlocs_layout.addWidget(self._no_structure_label)
        side_layout = QVBoxLayout()
        widgets_layout.addLayout(side_layout)
        from chimerax.ui.options import OptionsPanel, BooleanOption
        panel = OptionsPanel(scrolled=False)
        side_layout.addWidget(panel, alignment=Qt.AlignHCenter|Qt.AlignBottom)
        self._show_hbonds_opt = BooleanOption("", None, self._hbonds_shown_change, attr_name="show_hbonds",
            settings=AltlocExplorerSettings(session, tool_name))
        self._show_hbonds_opt.widget.setText("Show H-bonds")
        panel.add_option(self._show_hbonds_opt)
        params_but = QPushButton("H-bond parameters...")
        params_but.clicked.connect(self._show_hbonds_dialog)
        side_layout.addWidget(params_but, alignment=Qt.AlignHCenter|Qt.AlignTop)
        self.hbond_params_window = tw.create_child_window("Altloc H-Bond Parameters", close_destroys=False)
        self._populate_hbond_params()
        self.hbond_params_window.manage(initially_hidden=True)

        self._structure_widget = None
        self._changes_handler = None
        self._button_lookup = {}
        #TODO: react to alt loc additions/subtractions

        tw.manage(placement='side')

    def delete(self):
        if self._changes_handler:
            self._changes_handler.remove()
        self._structure_button.destroy()
        super().delete()

    def _apply_hb_params(self, *, hide=True, residues=None):
        cmd_name, ignore, args = self.hb_gui.get_command()
        structure = self._structure_button.value
        if residues is None:
            residues = [r for r in structure.residues if r.alt_locs]
        if residues:
            from chimerax.atomic import concise_residue_spec
            spec = concise_residue_spec(self.session, residues)
            run(self.session, f'{cmd_name} {spec} & @@num_alt_locs>1 {args} name "{self.hbonds_group_name}" updateGroup true')
        if hide:
            self.hbond_params_window.shown = False

    def _atomic_changes(self, trig_name, struct_trig_data):
        struct, trig_data = struct_trig_data
        if trig_data.num_deleted_atoms() > 0:
            for r in self._button_lookup.keys():
                if r.deleted:
                    self._structure_change()
                    return
                for a in r.atoms:
                    if a.num_alt_locs > 1:
                        break
                else:
                    # all atoms with altlocs gone
                    self._structure_change()
                    return
        if "alt_loc changed" in trig_data.atom_reasons():
            all_loc = None
            if self._structure_widget:
                for i in range(self._all_layout.count()):
                    but = self._all_layout.itemAt(i).widget()
                    if but.isChecked():
                        all_loc = but.text()
                        break
            changed_residues = set()
            for r, but_map in self._button_lookup.items():
                r_al = r.alt_loc
                if r_al != all_loc and all_loc in but_map:
                    all_loc = None
                    # to uncheck all buttons, have to turn auto-exclusvity off tempoarily
                    for i in range(self._all_layout.count()):
                        but = self._all_layout.itemAt(i).widget()
                        but.setAutoExclusive(False)
                        but.setChecked(False)
                        but.setAutoExclusive(True)
                for al, but in but_map.items():
                    if but.isChecked() and al != r_al:
                        but.setChecked(False)
                        changed_residues.add(r)
                    elif not but.isChecked() and al == r_al:
                        but.setChecked(True)
                        changed_residues.add(r)
            if changed_residues and self._show_hbonds_opt.value:
                self._apply_hb_params(residues=changed_residues)

    def _hbonds_shown_change(self, opt):
        struct = self._structure_button.value
        if struct:
            if opt.value:
                self._apply_hb_params()
            elif self.hbonds_group_name in struct.pbg_map:
                run(self.session, "close %s" % struct.pbg_map[self.hbonds_group_name].atomspec)

    def _make_structure_widget(self, structure):
        scroll_area = QScrollArea()
        from Qt.QtWidgets import QSizePolicy
        scroll_area.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding))
        widget = QWidget()
        layout = QGridLayout()
        layout.setSpacing(2)
        widget.setLayout(layout)
        from itertools import count
        rows = count()
        from chimerax.core.commands import run
        self._button_groups = []
        alt_loc_rs = [r for r in structure.residues if r.alt_locs]
        col_offset = 0
        letters = set()
        from math import sqrt, ceil
        needed_rows = len(alt_loc_rs) + 2
        num_columns = max(1, round(sqrt((len(alt_loc_rs)+2)/5)))
        # divide such that all but the last column have the same number of entries,
        # and the last column has the same or fewer than the earlier columns
        rows_per_column = ceil(needed_rows / num_columns)
        if num_columns > 1:
            for rpc in range(rows_per_column-1, 1, -1):
                last = needed_rows - rpc * num_columns
                if last <= rpc:
                    rows_per_column = rpc
        for r in alt_loc_rs:
            row = next(rows)
            button = QPushButton(r.string(omit_structure=True))
            button.clicked.connect(lambda *args, ses=self.session, run=run, spec=r.atomspec:
                run(ses, "show %s; view %s" % (spec, spec)))
            layout.addWidget(button, row, 0 + col_offset, alignment=Qt.AlignRight)
            button_group = QButtonGroup()
            self._button_groups.append(button_group)
            but_layout = QHBoxLayout()
            layout.addLayout(but_layout, row, 1 + col_offset, alignment=Qt.AlignLeft)
            for alt_loc in sorted(list(r.alt_locs)):
                letters.add(alt_loc)
                but = QRadioButton(alt_loc)
                self._button_lookup.setdefault(r, {})[alt_loc] = but
                but.setChecked(r.alt_loc == alt_loc)
                but.clicked.connect(lambda *args, ses=self.session, run=run, spec=r.atomspec, loc=alt_loc:
                    run(ses, "altlocs change %s %s" % (loc, spec)))
                but.clicked.connect(lambda *args, self=self, r=r: self._show_hbonds_opt.value
                    and self._apply_hb_params(residues=[r]))
                button_group.addButton(but)
                but_layout.addWidget(but, alignment=Qt.AlignCenter)
            if row == rows_per_column-1:
                layout.setColumnStretch(2+col_offset, 1)
                layout.setColumnMinimumWidth(2+col_offset, 5)
                col_offset += 3
                rows = count()
        row = next(rows)
        row = next(rows)
        if row >= rows_per_column:
            layout.setColumnStretch(2+col_offset, 1)
            layout.setColumnMinimumWidth(2+col_offset, 5)
            col_offset += 3
            row = 0
            layout.setColumnStretch(2+col_offset, 1)
            layout.setColumnMinimumWidth(2+col_offset, 5)
        layout.addWidget(QLabel("Set all to: "), row, 0 + col_offset, alignment=Qt.AlignRight)
        self._all_layout = QHBoxLayout()
        layout.addLayout(self._all_layout, row, 1 + col_offset, alignment=Qt.AlignLeft)
        from chimerax.core.commands import concise_model_spec
        for let in sorted(list(letters)):
            but = QRadioButton(let)
            but.clicked.connect(lambda *args, ses=self.session, run=run, s=structure, loc=let,
                concise=concise_model_spec:
                run(ses, "altlocs change %s %s" % (loc, concise(ses, [s], relevant_types=s.__class__))))
            self._all_layout.addWidget(but, alignment=Qt.AlignCenter)

        if not alt_loc_rs:
            layout.addWidget(QLabel("No alternate locations in this structure"), 0, 0)
        scroll_area.setWidget(widget)
        return scroll_area

    def _populate_hbond_params(self):
        layout = QVBoxLayout()
        self.hbond_params_window.ui_area.setLayout(layout)
        from chimerax.hbonds.gui import HBondsGUI
        self.hb_gui = HBondsGUI(self.session, settings_name="Altloc Explorer H-bonds", compact=True,
            inter_model=False, show_bond_restrict=False, show_inter_model=False, show_intra_model=False,
            show_intra_mol=False, show_intra_res=False, show_log=False, show_model_restrict=False,
            show_pseudobond_creation = False, show_retain_current=False, show_reveal=False,
            show_salt_only=False, show_save_file=False, show_select=False)
        layout.addWidget(self.hb_gui)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        self.bbox = bbox = qbbox(qbbox.Ok | qbbox.Apply | qbbox.Close | qbbox.Help)
        bbox.accepted.connect(self._apply_hb_params)
        bbox.button(qbbox.Apply).clicked.connect(lambda *args, s=self: s._apply_hb_params(hide=False))
        bbox.rejected.connect(lambda *args, tw=self.hbond_params_window: setattr(tw, 'shown', False))
        from chimerax.core.commands import run
        bbox.helpRequested.connect(lambda *, run=run, ses=self.session:
            run(ses, "help " + self.help + "#hbonds"))
        layout.addWidget(bbox)

    def _show_hbonds_dialog(self):
        self.hbond_params_window.shown =True

    def _structure_change(self):
        if self._structure_widget:
            self._altlocs_layout.removeWidget(self._structure_widget)
            self._structure_widget.hide()
            self._structure_widget.destroy()
            self._button_groups.clear()
            self._changes_handler.remove()
            self._changes_handler = None
            self._button_lookup.clear()

        structure = self._structure_button.value
        if structure:
            self._no_structure_label.hide()
            self._structure_widget = self._make_structure_widget(structure)
            self._altlocs_layout.addWidget(self._structure_widget)
            from chimerax.atomic import get_triggers
            self._changes_handler = structure.triggers.add_handler('changes', self._atomic_changes)
            self._hbonds_shown_change(self._show_hbonds_opt)
        else:
            self._no_structure_label.show()
            self._structure_widget = None

