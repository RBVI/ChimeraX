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

from chimerax.core.tools import ToolInstance
from chimerax.core.errors import UserError
from chimerax.atomic import AtomicStructure

_prd = None
def prep_rotamers_dialog(session, rotamers_tool_name):
    global _prd
    if _prd is None:
        _prd = PrepRotamersDialog(session, rotamers_tool_name)
    return _prd

class PrepRotamersDialog(ToolInstance):

    help = "help:user/tools/rotamers.html"
    SESSION_SAVE = False

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)
        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        tw.title = "Choose Rotamer Parameters"
        parent = tw.ui_area
        from Qt.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QGroupBox
        from Qt.QtCore import Qt
        self.layout = layout = QVBoxLayout()
        parent.setLayout(layout)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(QLabel("Show rotamers for selected residues..."), alignment=Qt.AlignCenter)

        self.rot_lib_option = session.rotamers.library_name_option()("Rotamer library", None,
            self._lib_change_cb)

        from chimerax.ui.options import EnumOption, OptionsPanel
        self.rot_lib = session.rotamers.library(self.rot_lib_option.value)
        res_name_list = self.lib_res_list()
        class ResTypeOption(EnumOption):
            values = res_name_list
        def_res_type = self._sel_res_type() or res_name_list[0]
        self.res_type_option = ResTypeOption("Rotamer type", def_res_type, None)

        opts = OptionsPanel(scrolled=False, contents_margins=(0,0,3,3))
        opts.add_option(self.res_type_option)
        opts.add_option(self.rot_lib_option)
        layout.addWidget(opts, alignment=Qt.AlignCenter)

        self.lib_description = QLabel(session.rotamers.description(self.rot_lib.name))
        layout.addWidget(self.lib_description, alignment=Qt.AlignCenter)

        self.rot_description_box = QGroupBox("Unusual rotamer codes")
        self.rot_description = QLabel("")
        box_layout = QVBoxLayout()
        box_layout.setContentsMargins(10,0,10,0)
        box_layout.addWidget(self.rot_description)
        self.rot_description_box.setLayout(box_layout)
        layout.addWidget(self.rot_description_box, alignment=Qt.AlignCenter)
        self._update_rot_description()

        self.citation_widgets = {}
        cw = self.citation_widgets[self.rot_lib_option.value] = self.citation_widgets['showing'] \
            = self._make_citation_widget()
        if cw:
            layout.addWidget(cw, alignment=Qt.AlignCenter)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        self.bbox = bbox = qbbox(qbbox.Ok | qbbox.Apply | qbbox.Close | qbbox.Help)
        bbox.accepted.connect(self.launch_rotamers)
        bbox.button(qbbox.Apply).clicked.connect(self.launch_rotamers)
        bbox.accepted.connect(self.delete) # slots executed in the order they are connected
        bbox.rejected.connect(self.delete)
        from chimerax.core.commands import run
        bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help " + self.help))
        layout.addWidget(bbox)

        tw.manage(placement=None)

    def delete(self):
        global _prd
        _prd = None
        super().delete()

    def launch_rotamers(self):
        from chimerax.atomic import selected_residues
        sel_residues = selected_residues(self.session)
        if not sel_residues:
            raise UserError("No residues selected")
        num_sel = len(sel_residues)
        if num_sel > 10:
            from chimerax.ui.ask import ask
            confirm = ask(self.session, "You have %d residues selected, which could bring up %d"
                " rotamer dialogs\nContinue?" % (num_sel, num_sel), title="Many Rotamer Dialogs")
            if confirm == "no":
                return
        res_type = self.res_type_option.value
        from chimerax.core.commands import run, StringArg
        from chimerax.rotamers import NoResidueRotamersError
        lib_name = StringArg.unparse(self.rot_lib_option.value)
        try:
            run(self.session, "swapaa interactive sel %s rotLib %s" % (res_type, lib_name))
        except NoResidueRotamersError:
            run(self.session, "swapaa sel %s rotLib %s" % (res_type, lib_name))

    def lib_res_list(self):
        res_name_list = list(self.rot_lib.residue_names) + ["ALA", "GLY"]
        res_name_list.sort()
        return res_name_list

    def _lib_change_cb(self, opt):
        cur_val = self.res_type_option.value
        self.rot_lib = self.session.rotamers.library(self.rot_lib_option.value)
        self.res_type_option.values = self.lib_res_list()
        self.res_type_option.remake_menu()
        if cur_val not in self.res_type_option.values:
            self.res_type_option.value = self._sel_res_type() or self.res_type_option.values[0]
        self.lib_description.setText(self.session.rotamers.description(self.rot_lib.name))
        prev_cite = self.citation_widgets['showing']
        if prev_cite:
            self.layout.removeWidget(prev_cite)
            prev_cite.hide()
        if self.rot_lib_option.value not in self.citation_widgets:
            self.citation_widgets[self.rot_lib_option.value] = self._make_citation_widget()
        new_cite = self.citation_widgets[self.rot_lib_option.value]
        if new_cite:
            from Qt.QtCore import Qt
            self.layout.insertWidget(self.layout.indexOf(self.bbox), new_cite, alignment=Qt.AlignCenter)
            new_cite.show()
        self.citation_widgets['showing'] = new_cite
        self._update_rot_description()
        self.tool_window.shrink_to_fit()

    def _make_citation_widget(self):
        if not self.rot_lib.citation:
            return None
        from chimerax.ui.widgets import Citation
        return Citation(self.session, self.rot_lib.citation, prefix="Publications using %s rotamers should"
            " cite:" % self.session.rotamers.ui_name(self.rot_lib.name),
            pubmed_id=self.rot_lib.cite_pubmed_id)

    def _sel_res_type(self):
        from chimerax.atomic import selected_residues
        sel_residues = selected_residues(self.session)
        sel_res_types = set([r.name for r in sel_residues])
        if len(sel_res_types) == 1:
            return self.rot_lib.map_res_name(sel_res_types.pop(), exemplar=sel_residues[0])
        return None

    def _update_rot_description(self):
        from chimerax.rotamers import RotamerLibrary
        unusual_info = []
        std_descriptions = RotamerLibrary.std_rotamer_res_descriptions
        for code, desc in self.rot_lib.res_name_descriptions.items():
            if code not in std_descriptions or std_descriptions[code] != desc:
                unusual_info.append((code, desc))
        if unusual_info:
            unusual_info.sort()
            self.rot_description.setText("\n".join(["%s: %s" % (code, desc) for code, desc in unusual_info]))
            self.rot_description_box.show()
        else:
            self.rot_description_box.hide()

_settings = None

class RotamerDialog(ToolInstance):

    help = "help:user/tools/rotamers.html"
    SESSION_SAVE = True
    registerer = "swap_res RotamerDialog"

    def __init__(self, session, tool_name, *args):
        ToolInstance.__init__(self, session, tool_name)
        if args:
            # being called directly rather than during session restore
            self.finalize_init(*args)

    def finalize_init(self, mgr, res_type, rot_lib_name, *, table_info=None):
        self.mgr = mgr
        self.res_type = res_type
        self.rot_lib_name = rot_lib_name

        self.subdialogs = {}
        from collections import OrderedDict
        self.opt_columns = OrderedDict()
        self.handlers = [
            self.mgr.triggers.add_handler('fewer rotamers', self._fewer_rots_cb),
            self.mgr.triggers.add_handler('self destroyed', self._mgr_destroyed_cb),
        ]
        from chimerax.ui.widgets import ItemTable
        global _settings
        if _settings is None:
            from chimerax.core.settings import Settings
            class _RotamerSettings(Settings):
                EXPLICIT_SAVE = { ItemTable.DEFAULT_SETTINGS_ATTR: {} }
            _settings = _RotamerSettings(self.session, "Rotamers")
        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area
        from Qt.QtWidgets import QVBoxLayout, QLabel, QCheckBox, QGroupBox, QWidget, QHBoxLayout, \
            QPushButton, QRadioButton, QButtonGroup, QGridLayout
        from Qt.QtCore import Qt
        self.layout = layout = QVBoxLayout()
        parent.setLayout(layout)
        lib_display_name = self.session.rotamers.ui_name(rot_lib_name)
        layout.addWidget(QLabel("%s %s rotamers" % (lib_display_name, res_type)))
        column_disp_widget = QWidget()
        class RotamerTable(ItemTable):
            def sizeHint(self):
                from Qt.QtCore import QSize
                return QSize(350, 450)
        self.table = RotamerTable(
            column_control_info=(column_disp_widget, _settings, {}, True, None, None, False),
            auto_multiline_headers=False)
        for i in range(len(self.mgr.rotamers[0].chis)):
            self.table.add_column("Chi %d" % (i+1), lambda r, i=i: r.chis[i], format="%6.1f")
        self.table.add_column("Prevalence", "rotamer_prob", format="%.6f ")

        if table_info:
            table_state, additional_col_info = table_info
            for col_type, title, data_fetch, display_format in additional_col_info:
                AtomicStructure.register_attr(self.session, data_fetch, self.registerer,
                    attr_type=(int if data_fetch.startswith("num") else float))
                self.opt_columns[col_type] = self.table.add_column(title, data_fetch, format=display_format)
        else:
            table_state = None
        self.table.data = self.mgr.rotamers
        self.table.launch(session_info=table_state)
        if not table_info:
            self.table.sortByColumn(len(self.mgr.rotamers[0].chis), Qt.DescendingOrder)
        self.table.selection_changed.connect(self._selection_change)
        layout.addWidget(self.table)
        if mgr.base_residue.name == res_type:
            self.retain_side_chain = QCheckBox("Retain original side chain")
            self.retain_side_chain.setChecked(False)
            layout.addWidget(self.retain_side_chain)
        else:
            self.retain_side_chain = None

        column_group = QGroupBox("Column display")
        layout.addWidget(column_group)
        cg_layout = QVBoxLayout()
        cg_layout.setContentsMargins(0,0,0,0)
        cg_layout.setSpacing(0)
        column_group.setLayout(cg_layout)
        cg_layout.addWidget(column_disp_widget)

        add_col_layout = QGridLayout()
        add_col_layout.setContentsMargins(0,0,0,0)
        add_col_layout.setSpacing(0)
        cg_layout.addLayout(add_col_layout)
        self.add_col_button = QPushButton("Calculate")
        add_col_layout.addWidget(self.add_col_button, 0, 0, alignment=Qt.AlignRight)
        radio_layout = QVBoxLayout()
        radio_layout.setContentsMargins(0,0,0,0)
        add_col_layout.addLayout(radio_layout, 0, 1, alignment=Qt.AlignLeft)
        self.button_group = QButtonGroup()
        self.add_col_button.clicked.connect(lambda checked, *, bg=self.button_group:
            self._show_subdialog(bg.checkedButton().text()))
        for add_type in ["H-Bonds", "Clashes", "Density"]:
            rb = QRadioButton(add_type)
            rb.clicked.connect(self._update_button_text)
            radio_layout.addWidget(rb)
            if not self.button_group.buttons():
                rb.setChecked(True)
            self.button_group.addButton(rb)
        self.ignore_solvent_button = QCheckBox("Ignore solvent")
        self.ignore_solvent_button.setChecked(True)
        add_col_layout.addWidget(self.ignore_solvent_button, 1, 0, 1, 2, alignment=Qt.AlignCenter)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        bbox = qbbox(qbbox.Cancel | qbbox.Help)
        bbox.addButton("Use Chosen Rotamer(s)", qbbox.AcceptRole)
        bbox.accepted.connect(self._apply_rotamer)
        bbox.rejected.connect(self.tool_window.destroy)
        from chimerax.core.commands import run
        bbox.helpRequested.connect(lambda *, run=run, ses=self.session: run(ses, "help " + self.help))
        layout.addWidget(bbox)
        self.tool_window.manage(placement=None)

    def delete(self, from_mgr=False):
        for handler in self.handlers:
            handler.remove()
        self.subdialogs.clear()
        self.opt_columns.clear()
        if not from_mgr:
            self.mgr.destroy()
        super().delete()

    @classmethod
    def restore_snapshot(cls, session, data):
        inst = super().restore_snapshot(session, data['ToolInstance'])
        if "rot_lib_name" in data:
            lib_name = data['rot_lib_name']
            lib_names = session.rotamers.library_names(installed_only=True)
            ui_name = session.rotamers.ui_name(lib_name)
        else:
            lib_name = ui_name = data['lib_display_name']
            lib_names = session.rotamers.library_names(installed_only=True, for_display=True)
        if lib_name not in lib_names:
            raise RuntimeError("Cannot restore Rotamers tool because %s rotamer library is not installed"
                % ui_name)
        inst.finalize_init(data['mgr'], data['res_type'], session.rotamers.library(lib_name).name,
            table_info=data['table info'])
        return inst

    def take_snapshot(self, session, flags):
        data = {
            'ToolInstance': ToolInstance.take_snapshot(self, session, flags),
            'mgr': self.mgr,
            'res_type': self.res_type,
            'rot_lib_name': self.rot_lib_name,
            'table info': (self.table.session_info(), [(col_type, c.title, c.data_fetch, c.display_format)
                for col_type, c in self.opt_columns.items()])
        }
        return data

    def _apply_rotamer(self):
        rots = self.table.selected
        if not rots:
            if self.button_group.checkedButton().text() in self.opt_columns:
                raise UserError("No rotamers selected")
            else:
                raise UserError("No rotamers selected.  Click the 'Calculate' button (not 'OK') to"
                    " add a column to the table.")
        rot_nums = [r.id[-1] for r in rots]
        from chimerax.core.commands import run
        cmd = "swapaa %s %s criteria %s rotLib %s" % (
            self.mgr.base_residue.string(style="command"),
            self.res_type,
            ",".join(["%d" % rn for rn in rot_nums]),
            self.rot_lib_name,
        )
        if self.retain_side_chain:
            cmd += " retain %s" % str(self.retain_side_chain.isChecked()).lower()
        run(self.session, cmd)
        self.delete()

    def _eval_vol(self, vol):
        from chimerax.core.utils import round_off
        AtomicStructure.register_attr(self.session, "sum_density", self.registerer, attr_type=float)
        for rot in self.mgr.rotamers:
            values = vol.interpolated_values(rot.atoms.coords, point_xform=rot.scene_position)
            total = 0
            for a, val in zip(rot.atoms, values):
                # 'is_side_chain' only works for actual polymers
                if a.name not in a.residue.aa_max_backbone_names:
                    total += val
            rot.sum_density = round_off(total, 3)
        sd_type = "Density"
        if sd_type in self.opt_columns:
            self.table.update_column(self.opt_columns[sd_type], data=True)
        else:
            self.opt_columns[sd_type] = self.table.add_column(sd_type, "sum_density", format="%g")

    def _fewer_rots_cb(self, trig_name, mgr):
        self.rot_table.set_data(self.mgr.rotamers)

    def _mgr_destroyed_cb(self, trig_name, mgr):
        self.delete(from_mgr=True)

    def _selection_change(self, selected, deselected):
        if self.table.selected:
            display = set(self.table.selected)
        else:
            display = set(self.mgr.rotamers)
        for rot in self.mgr.rotamers:
            rot.display = rot in display

    def _process_subdialog(self, sd_type):
        sd = self.subdialogs[sd_type]
        from chimerax.core.commands import run
        if sd_type == "H-Bonds":
            cmd_name, spec, args = sd.hbonds_gui.get_command()
            res = self.mgr.base_residue
            base_spec = "#!%s & ~%s" % (res.structure.id_string, res.string(style="command"))
            if self.ignore_solvent_button.isChecked():
                base_spec += " & ~solvent"
            hbs = run(self.session, "%s %s %s restrict #%s & ~@c,ca,n" %
                    (cmd_name, base_spec, args, self.mgr.group.id_string))
            AtomicStructure.register_attr(self.session, "num_hbonds", self.registerer, attr_type=int)
            for rotamer in self.mgr.rotamers:
                rotamer.num_hbonds = 0
            for d, a in hbs:
                if d.structure == res.structure:
                    a.structure.num_hbonds += 1
                else:
                    d.structure.num_hbonds += 1
            if sd_type in self.opt_columns:
                self.table.update_column(self.opt_columns[sd_type], data=True)
            else:
                self.opt_columns[sd_type] = self.table.add_column(sd_type, "num_hbonds", format="%d")
        elif sd_type == "Clashes":
            cmd_name, spec, args = sd.clashes_gui.get_command()
            res = self.mgr.base_residue
            base_spec = "#!%s & ~%s" % (res.structure.id_string, res.string(style="command"))
            if self.ignore_solvent_button.isChecked():
                base_spec += " & ~solvent"
            clashes = run(self.session, "%s %s %s restrict #%s & ~@c,ca,n" %
                    (cmd_name, base_spec, args, self.mgr.group.id_string))
            AtomicStructure.register_attr(self.session, "num_clashes", self.registerer, attr_type=int)
            for rotamer in self.mgr.rotamers:
                rotamer.num_clashes = 0
            for a, clashing in clashes.items():
                if a.structure != res.structure:
                    a.structure.num_clashes += len(clashing)
            if sd_type in self.opt_columns:
                self.table.update_column(self.opt_columns[sd_type], data=True)
            else:
                self.opt_columns[sd_type] = self.table.add_column(sd_type, "num_clashes", format="%d")
        else: # Density
            vol = sd.vol_list.value
            if not vol:
                return
            self._eval_vol(vol)
        self._update_button_text()

    def _show_subdialog(self, sd_type):
        if sd_type == "Density":
            from chimerax.map import Volume
            volumes = [m for m in self.session.models if isinstance(m, Volume)]
            if not volumes:
                raise UserError("Must open a volume/map file first!")
            if len(volumes) == 1:
                self._eval_vol(volumes[0])
                return
        if sd_type not in self.subdialogs:
            self.subdialogs[sd_type] = sd = self.tool_window.create_child_window("Add %s Column" % sd_type,
                close_destroys=False)
            from Qt.QtWidgets import QVBoxLayout, QDialogButtonBox as qbbox
            layout = QVBoxLayout()
            sd.ui_area.setLayout(layout)
            if sd_type == "H-Bonds":
                from chimerax.hbonds.gui import HBondsGUI
                sd.hbonds_gui = gui = HBondsGUI(self.session, settings_name="rotamers", reveal=True,
                    show_inter_model=False, show_intra_model=False, show_intra_mol=False,
                    show_intra_res=False, show_model_restrict=False, show_bond_restrict=False,
                    show_save_file=False)
                layout.addWidget(sd.hbonds_gui)
            elif sd_type == "Clashes":
                from chimerax.clashes.gui import ClashesGUI
                sd.clashes_gui = gui = ClashesGUI(self.session, False, settings_name="rotamers",
                    radius=0.075, show_restrict=False, show_bond_separation=False, show_res_separation=False,
                    show_inter_model=False, show_intra_model=False, show_intra_res=False,
                    show_intra_mol=False, show_attr_name=False, show_set_attrs=False,
                    show_checking_frequency=False, restrict="cross", bond_separation=0, reveal=True,
                    show_save_file=False)
                layout.addWidget(sd.clashes_gui)
            else: # Density
                gui = None
                from chimerax.ui.widgets import ModelListWidget
                from Qt.QtWidgets import QFormLayout
                density_layout = QFormLayout()
                layout.addLayout(density_layout)
                sd.vol_list = ModelListWidget(self.session, selection_mode='single', class_filter=Volume)
                density_layout.addRow("Select density:", sd.vol_list)
            bbox = qbbox(qbbox.Ok | qbbox.Close | qbbox.Help)
            bbox.accepted.connect(lambda *, sdt=sd_type: self._process_subdialog(sdt))
            bbox.accepted.connect(lambda *, sd=sd: setattr(sd, 'shown', False))
            bbox.rejected.connect(lambda *, sd=sd: setattr(sd, 'shown', False))
            from chimerax.core.commands import run
            bbox.helpRequested.connect(lambda *, run=run, ses=self.session:
                run(ses, "help help:user/tools/rotamers.html#evaluation"))
            if gui:
                reset_button = bbox.addButton("Reset", qbbox.ActionRole)
                reset_button.setToolTip("Reset to initial-installation defaults")
                reset_button.clicked.connect(lambda *args, gui=gui: gui.reset())
            layout.addWidget(bbox)
            sd.manage(placement=None)
        else:
            self.subdialogs[sd_type].title = "Update %s Column" % sd_type
        self.subdialogs[sd_type].shown = True

    def _update_button_text(self):
        cur_choice = self.button_group.checkedButton().text()
        if cur_choice.startswith("Density"):
            self.ignore_solvent_button.hide()
        else:
            self.ignore_solvent_button.show()
