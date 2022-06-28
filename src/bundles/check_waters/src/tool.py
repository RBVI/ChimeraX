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
        from Qt.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel
        from Qt.QtCore import Qt
        layout = QVBoxLayout()
        parent.setLayout(layout)
        layout.setContentsMargins(0,0,0,0)
        check_layout = QHBoxLayout()
        layout.addLayout(check_layout)
        check_layout.addWidget(QLabel("Check waters in:"), alignment=Qt.AlignRight)
        from chimerax.atomic.widgets import AtomicStructureMenuButton
        self.structure_menu = AtomicStructureMenuButton(session)
        check_layout.addWidget(self.structure_menu, alignment=Qt.AlignLeft)

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
        CheckWaterViewer(self.session, "Check Waters", s)

class CheckWaterViewer(ToolInstance):
    def __init__(self, session, tool_name, check_model=None, *, compare_info=None, model_labels=None):
        # if 'check_model' is None, we are being restored from a session 
        # and _finalize_init() will be called later
        super().__init__(session, tool_name)
        self.settings = CheckWaterSettings(session, tool_name)
        if check_model is None:
            return
        self._finalize_init(check_model, compare_info, model_labels)

    def _finalize_init(self, check_model, compare_info, model_labels, *, from_session=False):
        self.check_model = check_model
        self.compare_info = compare_info
        self.model_labels = model_labels
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
        self.handlers = [self.session.triggers.add_handler(REMOVE_MODELS, self._models_removed_cb)]

        # change any sphere representations into stick
        check_atoms = self.check_model.atoms
        check_spheres = check_atoms.filter(check_atoms.draw_modes == check_atoms.SPHERE_STYLE)
        check_spheres.draw_modes = check_atoms.STICK_STYLE
        if self.compare_model:
            compare_atoms = self.compare_model.atoms
            compare_spheres = compare_atoms.filter(compare_atoms.draw_modes == compare_atoms.SPHERE_STYLE)
            compare_spheres.draw_modes = compare_atoms.STICK_STYLE
        from chimerax.ui import MainToolWindow
        self.tool_window = MainToolWindow(self, close_destroys=False, statusbar=False)
        parent = self.tool_window.ui_area

        from Qt.QtWidgets import QHBoxLayout, QButtonGroup, QVBoxLayout, QRadioButton, QCheckBox
        from Qt.QtWidgets import QPushButton, QLabel, QToolButton
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        parent.setLayout(layout)
        if self.compared_waters:
            # we kept the input waters
            self.radio_group = QButtonGroup(parent)
            self.radio_group.buttonClicked.connect(self._update_residues)
            self.button_layout = but_layout = QVBoxLayout()
            layout.addLayout(but_layout)
            self.after_only_button = QRadioButton(parent)
            self.radio_group.addButton(self.after_only_button)
            but_layout.addWidget(self.after_only_button)
            self.in_common_button = QRadioButton(parent)
            self.radio_group.addButton(self.in_common_button)
            but_layout.addWidget(self.in_common_button)
            self.before_only_button = QRadioButton(parent)
            self.radio_group.addButton(self.before_only_button)
            but_layout.addWidget(self.before_only_button)
            self._update_button_texts()
            self.after_only_button.setChecked(True)
            self.filter_residues = self.compared_waters[1]
        else:
            # didn't keep the input waters
            self.radio_group = None
            from .compare import _water_residues
            self.filter_residues = sorted(_water_residues(self.check_model))
        self.filter_model = self.check_model
        from chimerax.atomic.widgets import ResidueListWidget
        self.res_list = ResidueListWidget(self.session, filter_func=self._filter_residues)
        if from_session:
            # Complicated code to avoid having the residue list callback change
            # the restored session state:  after one frame drawn look for a new
            # frame where there aren't changes pending, then hook up callback.
            # Three handlers are involved because "frame drawn" is not called
            # if "new frame" has nothing to draw.
            cb_info = []
            def new_frame_handler(*args, ses=self.session, cb_info=cb_info,
                    signal=self.res_list.value_changed, handler=self._res_sel_cb):
                if ses.change_tracker.changed:
                    from chimerax.atomic import get_triggers
                    get_triggers().add_handler("changes done", cb_info[0])
                else:
                    signal.connect(handler)
                from chimerax.core.triggerset import DEREGISTER
                return DEREGISTER
            def changes_done_handler(*args, ses=self.session):
                ses.triggers.add_handler("new frame", new_frame_handler)
                from chimerax.core.triggerset import DEREGISTER
                return DEREGISTER
            cb_info.append(changes_done_handler)
            def frame_drawn_handler(*args, ses=self.session):
                ses.triggers.add_handler("new frame", new_frame_handler)
                from chimerax.core.triggerset import DEREGISTER
                return DEREGISTER
            self.session.triggers.add_handler("frame drawn", frame_drawn_handler)
        else:
            self.res_list.value_changed.connect(self._res_sel_cb)
        layout.addWidget(self.res_list)

        self.hbond_groups = {}
        controls_layout = QVBoxLayout()
        hbonds_layout = QVBoxLayout()
        hbonds_layout.setSpacing(1)
        self.show_hbonds = check = QCheckBox("Show hydrogen bonds")
        check.setChecked(self.settings.show_hbonds)
        check.clicked.connect(self._show_hbonds_cb)
        hbonds_layout.addWidget(check)
        disclosure_layout = QHBoxLayout()
        self.params_arrow = QToolButton()
        self.params_arrow.setArrowType(Qt.RightArrow)
        self.params_arrow.setMaximumSize(16, 16)
        self.params_arrow.clicked.connect(self._hb_disclosure_cb)
        disclosure_layout.addWidget(self.params_arrow, alignment=Qt.AlignRight)
        disclosure_layout.addWidget(QLabel(" H-Bond Parameters"), alignment=Qt.AlignLeft)
        disclosure_layout.addStretch(1)
        hbonds_layout.addLayout(disclosure_layout)
        from chimerax.hbonds.gui import HBondsGUI
        self.hb_gui = HBondsGUI(self.session, settings_name="CheckWater H-bonds", compact=True, inter_model=False,
            show_bond_restrict=False, show_inter_model=False, show_intra_model=False, show_intra_mol=False,
            show_intra_res=False, show_log=False, show_model_restrict=False, show_retain_current=False,
            show_reveal=False, show_salt_only=False, show_save_file=False, show_select=False)
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
        if not from_session and self.settings.show_hbonds:
            self._show_hbonds_cb(True)
        delete_layout = QHBoxLayout()
        but = QPushButton("Delete")
        but.clicked.connect(self._delete_waters)
        delete_layout.addWidget(but, alignment=Qt.AlignRight)
        delete_layout.addWidget(QLabel("chosen water(s)"), alignment=Qt.AlignLeft)
        controls_layout.addLayout(delete_layout)
        layout.addLayout(controls_layout)

        self.tool_window.manage('side')

    def delete(self):
        for handler in self.handlers:
            handler.remove()
        self.compare_model = self.check_model = None
        super().delete()

    @classmethod
    def restore_snapshot(cls, session, data):
        inst = super().restore_snapshot(session, data['ToolInstance'])
        inst._finalize_init(data['check_model'], data['compare_info'], data['model_labels'],
            from_session=True)
        if data['radio info']:
            for but in inst.radio_group.buttons():
                if but.text() == data['radio info']:
                    but.setChecked(True)
                    inst._update_residues()
                    break
        if data['water']:
            inst.res_list.blockSignals(True)
            inst.res_list.value = data['water']
            inst.res_list.blockSignals(False)
        inst.settings.show_hbonds = data['show hbonds']
        inst.show_hbonds.setChecked(data['show hbonds'])
        return inst

    SESSION_SAVE = True

    def take_snapshot(self, session, flags):
        data = {
            'ToolInstance': ToolInstance.take_snapshot(self, session, flags),
            'compared_waters': self.compared_waters,
            'check_model': self.check_model,
            'compare_info': self.compare_info,
            'model_labels': self.model_labels,
            'radio info': self.radio_group.checkedButton().text() if self.radio_group else None,
            'show hbonds': self.settings.show_hbonds,
            'version': 1,
            'water': self.res_list.value,
        }
        return data

    def _delete_waters(self):
        waters = self.res_list.value
        if not waters:
            raise UserError("No waters chosen")
        if len(waters) > 1:
            from chimerax.ui.ask import ask
            if ask(self.session, "Really delete %d waters?" % len(waters),
                    default="no", title="Delete waters") == "no":
                return
        else:
            # go to the next water in the list
            all_values = self.res_list.all_values
            if len(all_values) > 1:
                next_row = (all_values.index(waters[0]) + 1) % len(all_values)
                self.res_list.value = [all_values[next_row]]
                self.res_list.scrollToItem(self.res_list.item(next_row))
        from chimerax.atomic import Residues
        Residues(waters).atoms.delete()

    def _filter_residues(self, r):
        return r.structure == self.filter_model and r in self.filter_residues

    def _hb_disclosure_cb(self, *args):
        is_hidden = self.hb_gui.isHidden()
        self.params_arrow.setArrowType(Qt.DownArrow if is_hidden else Qt.RightArrow)
        self.hb_gui.setHidden(not is_hidden)
        self.hb_apply_but.setHidden(not is_hidden)
        self.hb_apply_label.setHidden(not is_hidden)

    def _make_hb_group(self):
        model = self.filter_model
        if self.radio_group:
            all_input, after_only, douse_in_common, input_in_common = self.compared_waters
            checked_button = self.radio_group.checkedButton()
            text = checked_button.text()
            left_paren = text.index('(')
            name = text[:left_paren] + "water H-bonds"
            if checked_button == self.after_only_button:
                waters = after_only
            elif checked_button == self.before_only_button:
                waters = all_input - input_in_common
            else:
                waters = douse_in_common
        else:
            waters = self.filter_residues
            name = "water H-bonds"
        cmd_name, spec, args = self.hb_gui.get_command()
        from chimerax.atomic import concise_residue_spec
        spec = concise_residue_spec(self.session, waters)
        from chimerax.core.commands import run, StringArg
        run(self.session, '%s %s %s restrict any name %s' % (cmd_name, spec, args, StringArg.unparse(name)))
        if "showDist true" in args:
            run(self.session, 'label size 16')
            run(self.session, 'distance style decimalPlaces 2')
        return model.pseudobond_group(name, create_type="per coordset")

    def _models_removed_cb(self, trig_name, trig_data):
        if self.check_model in trig_data:
            self.delete()
        elif self.compare_model in trig_data:
            self.after_only_button.setChecked(True)
            self._update_residues()
            self.tool_window.ui_area.layout().removeItem(self.button_layout)

    def _res_sel_cb(self):
        selected = self.res_list.value
        if not selected:
            cmd = "~select; view %s" % self.check_model.atomspec
        else:
            if selected[0].structure.display:
                base_cmd = ""
            else:
                base_cmd = "show %s models; " % selected[0].structure.atomspec
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
            try:
                group = self.hbond_groups[group_key]
            except KeyError:
                self.hbond_groups[group_key] = group = self._make_hb_group()
            group.display = True

    def _update_button_texts(self):
        all_input, after_only, douse_in_common, input_in_common = self.compared_waters
        self.after_only_button.setText("%s only (%d)" % (self.check_label, len(after_only)))
        self.in_common_button.setText("In common (%d)" % len(input_in_common))
        self.before_only_button.setText("%s only (%d)" % (self.compare_label,
            len(all_input - input_in_common)))

    def _update_hbonds(self):
        group_key = self.radio_group.checkedButton() \
            if (self.radio_group and self.settings.show_hbonds) else None
        self.session.models.close([group for key, group in self.hbond_groups.items() if key != group_key])
        if self.settings.show_hbonds:
            self.hbond_groups[group_key] = self._make_hb_group()

    def _update_residues(self):
        all_input, after_only, douse_in_common, input_in_common = self.compared_waters
        checked = self.radio_group.checkedButton()
        if checked == self.after_only_button:
            self.filter_model = self.check_model
            self.filter_residues = after_only
        elif checked == self.before_only_button:
            self.filter_model = self.compare_model
            self.filter_residues = all_input - input_in_common
        else:
            self.filter_model = self.compare_model
            self.filter_residues = input_in_common
        if self.show_hbonds.isChecked():
            self._show_hbonds_cb(True)
        self.res_list.refresh()
