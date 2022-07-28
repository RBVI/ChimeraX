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
from chimerax.core.settings import Settings
from Qt.QtCore import Qt

class CheckWaterSettings(Settings):
    AUTO_SAVE = {
        "show_hbonds": True,
    }

class CheckWatersInputTool(ToolInstance):
    SESSION_ENDURING = True

    def __init__(self, session):
        super().__init__(session, "Check Waters Input")
        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self, close_destroys=False, statusbar=False)
        tw.title = "Choose Structure for Water Checking"
        parent = self.tool_window.ui_area
        parent = tw.ui_area
        from Qt.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QFormLayout
        from Qt.QtCore import Qt
        layout = QVBoxLayout()
        parent.setLayout(layout)
        check_layout = QFormLayout()
        layout.addLayout(check_layout)
        from chimerax.atomic.widgets import AtomicStructureMenuButton
        self.structure_menu = AtomicStructureMenuButton(session)
        check_layout.addRow("Check waters in:", self.structure_menu)
        from chimerax.map import Volume
        from chimerax.ui.widgets import ModelListWidget
        self.map_list = ModelListWidget(session, selection_mode='single', class_filter=Volume)
        check_layout.addRow("Against volume/map (optional):", self.map_list)


        from Qt.QtWidgets import QDialogButtonBox as qbbox
        self.bbox = bbox = qbbox(qbbox.Ok | qbbox.Apply | qbbox.Close | qbbox.Help)
        bbox.accepted.connect(self.launch_cw_tool)
        bbox.button(qbbox.Apply).clicked.connect(self.launch_cw_tool)
        bbox.accepted.connect(self.delete) # slots executed in the order they are connected
        bbox.rejected.connect(self.delete)
        from chimerax.core.commands import run
        bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help " + self.help))
        layout.addWidget(bbox)

        tw.manage(placement=None)

    def launch_cw_tool(self):
        s = self.structure_menu.value
        if not s:
            raise UserError("No structure chosen for checking")
        CheckWaterViewer(self.session, "Check Waters", s, compare_map=self.map_list.value)

class CheckWaterViewer(ToolInstance):

    help = "help:user/tools/checkwaters.html"

    DENSITY_ATTR = "cw_density"
    HB_COUNT_ATTR = "num_cw_hbonds"

    def __init__(self, session, tool_name, check_model=None, *, compare_info=None, model_labels=None,
            compare_map=None):
        # if 'check_model' is None, we are being restored from a session 
        # and _finalize_init() will be called later
        super().__init__(session, tool_name)
        self.settings = CheckWaterSettings(session, tool_name)
        if check_model is None:
            return
        self._finalize_init(check_model, compare_info, model_labels, compare_map)

    def _finalize_init(self, check_model, compare_info, model_labels, compare_map, *, session_info=None):
        self.check_model = check_model
        self.compare_info = compare_info
        self.model_labels = model_labels
        self.compare_map = compare_map
        if session_info:
            self.check_waters, self.hbond_groups, table_info = session_info
        else:
            table_info = None
        if compare_info is None:
            self.compare_model = self.compared_waters = None
        else:
            self.compare_model, self.compared_waters = compare_info
        if self.compared_waters is None and self.compare_model:
            from . import compare_waters
            self.compared_waters = compare_waters(compare_model, check_model)
        self.compared_waters = [x.__class__(sorted(x)) for x in self.compared_waters
            ] if self.compared_waters else None
        if model_labels is None:
            if self.compare_model:
                if self.compare_model.name == self.check_model.name:
                    self.compare_label = str(self.compare_model)
                    self.check_label = str(self.check_model)
                else:
                    self.compare_label = self.compare_model.name.capitalize()
                    self.check_label = self.check_model.name.capitalize()
            else:
                self.check_label = self.check_model.name.capitalize()
        else:
            self.compare_label, self.check_label = [x.capitalize() for x in model_labels]
        from chimerax.core.models import REMOVE_MODELS
        from chimerax.atomic import get_triggers
        self.handlers = [
            self.session.triggers.add_handler(REMOVE_MODELS, self._models_removed_cb),
            get_triggers().add_handler("changes", self._changes_cb)
        ]

        # change any sphere representations into stick
        if not session_info:
            check_atoms = self.check_model.atoms
            check_spheres = check_atoms.filter(check_atoms.draw_modes == check_atoms.SPHERE_STYLE)
            check_spheres.draw_modes = check_atoms.STICK_STYLE
            if self.compare_model:
                compare_atoms = self.compare_model.atoms
                compare_spheres = compare_atoms.filter(
                    compare_atoms.draw_modes == compare_atoms.SPHERE_STYLE)
                compare_spheres.draw_modes = compare_atoms.STICK_STYLE
        from chimerax.ui import MainToolWindow
        self.tool_window = MainToolWindow(self, close_destroys=False, statusbar=False)
        parent = self.tool_window.ui_area

        from Qt.QtWidgets import QHBoxLayout, QButtonGroup, QVBoxLayout, QRadioButton, QCheckBox
        from Qt.QtWidgets import QPushButton, QLabel, QToolButton, QGridLayout
        layout = QVBoxLayout()
        layout.setContentsMargins(2,2,2,2)
        layout.setSpacing(0)
        parent.setLayout(layout)
        if self.compared_waters:
            # we kept the input waters
            self.radio_group = QButtonGroup(parent)
            self.radio_group.buttonClicked.connect(self._update_residues)
            self.button_layout = but_layout = QHBoxLayout()
            layout.addLayout(but_layout)
            self.after_only_button = QRadioButton(parent)
            self.radio_group.addButton(self.after_only_button)
            but_layout.addWidget(self.after_only_button, alignment=Qt.AlignCenter)
            self.in_common_button = QRadioButton(parent)
            self.radio_group.addButton(self.in_common_button)
            but_layout.addWidget(self.in_common_button, alignment=Qt.AlignCenter)
            self.before_only_button = QRadioButton(parent)
            self.radio_group.addButton(self.before_only_button)
            but_layout.addWidget(self.before_only_button, alignment=Qt.AlignCenter)
            self._update_button_texts()
            self.after_only_button.setChecked(True)
            all_input, after_only, douse_in_common, input_in_common = self.compared_waters
            table_waters = after_only
        else:
            # didn't keep the input waters
            self.radio_group = None
            if not session_info:
                from .compare import _water_residues
                self.check_waters = sorted(_water_residues(self.check_model))
            table_waters = self.check_waters
        data_layout = QHBoxLayout()
        layout.addLayout(data_layout)
        from chimerax.ui.widgets import ItemTable
        self.res_table = ItemTable()
        self.res_table.add_column("Water", str)
        self.hbonds_column = self.res_table.add_column("H-bonds", self.HB_COUNT_ATTR)
        if self.compare_map:
            self._compute_densities()
            self.res_table.add_column("Density", self.DENSITY_ATTR, format="%g")
        self.res_table.selection_changed.connect(self._res_sel_cb)
        data_layout.addWidget(self.res_table)

        controls_layout = QVBoxLayout()
        hbonds_layout = QVBoxLayout()
        hbonds_layout.setSpacing(1)
        self.show_hbonds = check = QCheckBox("Show H-bonds")
        check.setChecked(self.settings.show_hbonds)
        check.clicked.connect(self._show_hbonds_cb)
        hbonds_layout.addWidget(check)
        disclosure_layout = QHBoxLayout()
        self.params_arrow = QToolButton()
        self.params_arrow.setArrowType(Qt.RightArrow)
        self.params_arrow.setMaximumSize(16, 16)
        self.params_arrow.clicked.connect(self._hb_disclosure_cb)
        disclosure_layout.addWidget(self.params_arrow, alignment=Qt.AlignRight)
        disclosure_layout.addWidget(QLabel(" H-bond parameters..."), alignment=Qt.AlignLeft)
        disclosure_layout.addStretch(1)
        hbonds_layout.addLayout(disclosure_layout)
        from chimerax.hbonds.gui import HBondsGUI
        self.hb_gui = HBondsGUI(self.session, settings_name="CheckWater H-bonds", compact=True,
            inter_model=False, show_bond_restrict=False, show_inter_model=False, show_intra_model=False,
            show_intra_mol=False, show_intra_res=False, show_log=False, show_model_restrict=False,
            show_pseudobond_creation = False, show_retain_current=False, show_reveal=False,
            show_salt_only=False, show_save_file=False, show_select=False)
        self.hb_gui.layout().setContentsMargins(0,0,0,0)
        self.hb_gui.setHidden(True)
        hbonds_layout.addWidget(self.hb_gui)
        hb_apply_layout = QHBoxLayout()
        hb_apply_layout.addStretch(1)
        self.hb_apply_but = apply_but = QPushButton("Apply")
        self.hb_apply_but.setHidden(True)
        apply_but.clicked.connect(self._update_hbonds)
        hb_apply_layout.addWidget(apply_but)
        self.hb_apply_label = QLabel("above parameters")
        self.hb_apply_label.setHidden(True)
        hb_apply_layout.addWidget(self.hb_apply_label)
        hb_apply_layout.addStretch(1)
        hbonds_layout.addLayout(hb_apply_layout)
        controls_layout.addLayout(hbonds_layout)
        delete_layout = QGridLayout()
        but = QPushButton("Delete")
        but.clicked.connect(self._delete_waters)
        delete_layout.addWidget(but, 0, 0, alignment=Qt.AlignRight)
        delete_layout.addWidget(QLabel(" chosen water(s)"), 0, 1, alignment=Qt.AlignLeft)
        self.next_after_del = QCheckBox("Go to next water in list after Delete")
        self.next_after_del.setChecked(True)
        delete_layout.addWidget(self.next_after_del, 1, 0, 1, 2, alignment=Qt.AlignCenter)
        controls_layout.addLayout(delete_layout)
        clip_layout = QHBoxLayout()
        self.unclip_button = QPushButton("Unclip")
        self.unclip_button.clicked.connect(self._unclip_cb)
        clip_layout.addWidget(self.unclip_button, alignment=Qt.AlignRight)
        clip_layout.addWidget(QLabel(" view"), alignment=Qt.AlignLeft)
        controls_layout.addLayout(clip_layout)
        data_layout.addLayout(controls_layout)
        # The H-bonds GUI needs to exist before running _make_hb_groups() and showing the
        # H-bonds in the table, so these lines are down here
        if not session_info:
            self._make_hb_groups()
        self.res_table.data = table_waters
        self.res_table.launch(session_info=table_info)
        if not session_info:
            self.res_table.sort_by(self.hbonds_column, self.res_table.SORT_DESCENDING)
            if self.settings.show_hbonds:
                self._show_hbonds_cb(True)
            self._selected_treatment(table_waters)

        self.tool_window.manage('side')

    def delete(self):
        for handler in self.handlers:
            handler.remove()
        self.compare_model = self.check_model = None
        super().delete()

    @classmethod
    def restore_snapshot(cls, session, data):
        inst = super().restore_snapshot(session, data['ToolInstance'])
        if data['version'] == 1:
            # Just can't initialize completely properly from version 1 info
            session_info = None
        else:
            session_info = (data['check_waters'], data['hbond_groups'], data['table info'])
        inst._finalize_init(data['check_model'], data['compare_info'], data['model_labels'],
            data.get('compare_map', None), session_info=session_info)
        if data['radio info']:
            for but in inst.radio_group.buttons():
                if but.text() == data['radio info']:
                    but.setChecked(True)
                    inst._update_residues()
                    break
        inst.settings.show_hbonds = data['show hbonds']
        inst.show_hbonds.setChecked(data['show hbonds'])
        return inst

    SESSION_SAVE = True

    def take_snapshot(self, session, flags):
        data = {
            'ToolInstance': ToolInstance.take_snapshot(self, session, flags),
            'compared_waters': self.compared_waters,
            'check_model': self.check_model,
            'check_waters': self.check_waters,
            'compare_info': self.compare_info,
            'compare_map': self.compare_map,
            'hbond_groups': self.hbond_groups,
            'model_labels': self.model_labels,
            'radio info': self.radio_group.checkedButton().text() if self.radio_group else None,
            'show hbonds': self.settings.show_hbonds,
            'table info': self.res_table.session_info(),
            'version': 2,
        }
        return data

    def _changes_cb(self, trig_name, changes):
        if changes.num_deleted_residues() > 0:
            cur_data = self.res_table.data
            live_data = [d for d in cur_data if not d.deleted]
            if len(live_data) < len(cur_data):
                self.res_table.data = live_data

    def _compute_densities(self):
        from chimerax.atomic import concise_residue_spec, Residue
        Residue.register_attr(self.session, self.DENSITY_ATTR, "Check Waters", attr_type=float)
        residue_groups = []
        if self.radio_group:
            all_input, after_only, douse_in_common, input_in_common = self.compared_waters
            residue_groups = [after_only, all_input - input_in_common, douse_in_common]
        else:
            residue_groups = [self.check_waters]
        for res_group in residue_groups:
            for r in res_group:
                setattr(r, self.DENSITY_ATTR, sum(self.compare_map.interpolated_values(r.atoms.coords,
                    point_xform=r.structure.scene_position)))

    def _delete_waters(self):
        waters = self.res_table.selected
        if not waters:
            raise UserError("No waters chosen")
        if len(waters) > 1:
            from chimerax.ui.ask import ask
            if ask(self.session, "Really delete %d waters?" % len(waters),
                    default="no", title="Delete waters") == "no":
                return
        else:
            # go to the next water in the list
            all_values = self.res_table.sorted_data
            if len(all_values) > 1 and self.next_after_del.isChecked():
                next_row = (all_values.index(waters[0]) + 1) % len(all_values)
                datum = all_values[next_row]
                self.res_table.selected = [datum]
                self.res_table.scroll_to(datum)
        # remove soon-to-be dead water(s) before table tries to redraw...
        self.res_table.data = [item for item in self.res_table.data if item not in waters]
        from chimerax.atomic import Residues
        Residues(waters).atoms.delete()

    def _hb_disclosure_cb(self, *args):
        is_hidden = self.hb_gui.isHidden()
        self.params_arrow.setArrowType(Qt.DownArrow if is_hidden else Qt.RightArrow)
        self.hb_gui.setHidden(not is_hidden)
        self.hb_apply_but.setHidden(not is_hidden)
        self.hb_apply_label.setHidden(not is_hidden)

    def _make_hb_groups(self):
        self.hbond_groups = {}
        input_data = []
        if self.radio_group:
            all_input, after_only, douse_in_common, input_in_common = self.compared_waters
            for but in self.radio_group.buttons():
                text = but.text()
                left_paren = text.index('(')
                name = text[:left_paren] + "water H-bonds"
                if but == self.after_only_button:
                    waters = after_only
                    model = self.check_model
                elif but == self.before_only_button:
                    waters = all_input - input_in_common
                    model = self.compare_model
                else:
                    waters = douse_in_common
                    model = self.check_model
                input_data.append((but, name, waters, model))
        else:
            input_data.append((None, "water H-bonds", self.check_waters, self.check_model))
        cmd_name, spec, args = self.hb_gui.get_command()
        from chimerax.atomic import concise_residue_spec, Residue
        Residue.register_attr(self.session, self.HB_COUNT_ATTR, "Check Waters", attr_type=int)
        for key, name, waters, model in input_data:
            spec = concise_residue_spec(self.session, waters)
            from chimerax.core.commands import run, StringArg
            hbonds = run(self.session, '%s %s %s restrict any name %s' % (cmd_name, spec, args,
                StringArg.unparse(name)))
            self.hbond_groups[key] = model.pseudobond_group(name, create_type="per coordset")
            water_set = set(waters)
            for water in waters:
                setattr(water, self.HB_COUNT_ATTR, 0)
            # avoid overcounting water pairs that can both donate and accept to each other
            seen = set()
            for hb in hbonds:
                if hb in seen:
                    continue
                for a in hb:
                    if a.residue in water_set:
                        setattr(a.residue, self.HB_COUNT_ATTR, getattr(a.residue, self.HB_COUNT_ATTR)+1)
                seen.add(hb)
                seen.add((hb[1], hb[0]))

    def _models_removed_cb(self, trig_name, trig_data):
        if self.check_model in trig_data:
            self.delete()
        elif self.compare_model in trig_data:
            self.after_only_button.setChecked(True)
            self._update_residues()
            self.tool_window.ui_area.layout().removeItem(self.button_layout)

    def _res_sel_cb(self, newly_selected, newly_deselected):
        self._selected_treatment(self.res_table.selected)

    def _selected_treatment(self, selected):
        if not selected:
            cmd = "~select"
        else:
            structure = selected[0].structure
            if structure.display:
                base_cmd = ""
            else:
                base_cmd = "show %s models; " % structure.atomspec
            if self.compare_model:
                other_model = self.compare_model if structure == self.check_model else self.check_model
                if other_model.display:
                    base_cmd += "hide %s models; " % other_model.atomspec
            from chimerax.atomic import concise_residue_spec
            spec = concise_residue_spec(self.session, selected)
            cmd = base_cmd + f"select {spec}; disp {spec} :<4; view {spec} @<4"
        from chimerax.core.commands import run
        run(self.session, cmd)

    def _show_hbonds_cb(self, checked):
        self.settings.show_hbonds = checked
        for group in self.hbond_groups.values():
            group.display = False
        if checked:
            group_key = self.radio_group.checkedButton() if self.radio_group else None
            self.hbond_groups[group_key].display = True

    def _unclip_cb(self):
        from chimerax.core.commands import run
        run(self.session, "clip off")

    def _update_button_texts(self):
        all_input, after_only, douse_in_common, input_in_common = self.compared_waters
        self.after_only_button.setText("%s only (%d)" % (self.check_label, len(after_only)))
        self.in_common_button.setText("In common (%d)" % len(input_in_common))
        self.before_only_button.setText("%s only (%d)" % (self.compare_label,
            len(all_input - input_in_common)))

    def _update_hbonds(self):
        self.session.models.close([group for group in self.hbond_groups.values()])
        self._make_hb_groups()
        if not self.settings.show_hbonds:
            for grp in self.hbond_groups.values():
                grp.display = False
        self.res_table.update_column(self.hbonds_column, data=True)

    def _update_residues(self):
        all_input, after_only, douse_in_common, input_in_common = self.compared_waters
        checked = self.radio_group.checkedButton()
        if checked == self.after_only_button:
            residues = after_only
        elif checked == self.before_only_button:
            residues = all_input - input_in_common
        else:
            residues = douse_in_common
        if self.show_hbonds.isChecked():
            self._show_hbonds_cb(True)
        self.res_table.data = residues
        self._selected_treatment(residues)
