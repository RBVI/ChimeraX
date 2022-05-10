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

class DouseResultsViewer(ToolInstance):
    def __init__(self, session, tool_name, orig_model=None, douse_model=None, compared_waters=None):
        # if 'model' is None, we are being restored from a session and _finalize_init() will be called later
        super().__init__(session, tool_name)
        if douse_model is None:
            return
        self._finalize_init(orig_model, douse_model, compared_waters)

    def _finalize_init(self, orig_model, douse_model, compared_waters, *, from_session=False):
        self.orig_model = orig_model
        self.douse_model = douse_model
        self.compared_waters = compared_waters
        from chimerax.core.models import REMOVE_MODELS
        self.handlers = [self.session.triggers.add_handler(REMOVE_MODELS, self._models_removed_cb)]

        from chimerax.ui import MainToolWindow
        self.tool_window = MainToolWindow(self, close_destroys=False, statusbar=False)
        parent = self.tool_window.ui_area

        from Qt.QtWidgets import QHBoxLayout, QButtonGroup, QVBoxLayout, QRadioButton
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        parent.setLayout(layout)
        if compared_waters:
            # we kept the input waters
            self.radio_group = QButtonGroup(parent)
            self.radio_group.buttonClicked.connect(self._update_residues)
            self.button_layout = but_layout = QVBoxLayout()
            layout.addLayout(but_layout)
            self.douse_only_button = QRadioButton(parent)
            self.radio_group.addButton(self.douse_only_button)
            but_layout.addWidget(self.douse_only_button)
            self.in_common_button = QRadioButton(parent)
            self.radio_group.addButton(self.in_common_button)
            but_layout.addWidget(self.in_common_button)
            self.original_only_button = QRadioButton(parent)
            self.radio_group.addButton(self.original_only_button)
            but_layout.addWidget(self.original_only_button)
            self._update_button_texts()
            self.douse_only_button.setChecked(True)
            self.filter_residues = compared_waters[1]
        else:
            # didn't keep the input waters
            self.radio_group = None
            from .douse import _water_residues
            self.filter_residues = _water_residues(self.douse_model)
        self.filter_model = self.douse_model
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

        self.tool_window.manage('side')

    def delete(self):
        for handler in self.handlers:
            handler.remove()
        self.orig_model = self.douse_model = None
        super().delete()

    @classmethod
    def restore_snapshot(cls, session, data):
        inst = super().restore_snapshot(session, data['ToolInstance'])
        inst._finalize_init(data['orig_model'], data['douse_model'], data['compared_waters'],
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
        return inst

    SESSION_SAVE = True

    def take_snapshot(self, session, flags):
        data = {
            'ToolInstance': ToolInstance.take_snapshot(self, session, flags),
            'compared_waters': self.compared_waters,
            'douse_model': self.douse_model,
            'orig_model': self.orig_model,
            'radio info': self.radio_group.checkedButton().text() if self.radio_group else None,
            'version': 1,
            'water': self.res_list.value,
        }
        return data

    def _filter_residues(self, r):
        return r.structure == self.filter_model and r in self.filter_residues

    def _models_removed_cb(self, trig_name, trig_data):
        if self.douse_model in trig_data:
            self.delete()
        elif self.orig_model in trig_data:
            self.douse_only_button.setChecked(True)
            self._update_residues()
            self.tool_window.ui_area.layout().removeItem(self.button_layout)

    def _res_sel_cb(self):
        selected = self.res_list.value
        if not selected:
            cmd = "~select; view %s" % self.douse_model.atomspec
        else:
            if selected[0].structure.display:
                base_cmd = ""
            else:
                base_cmd = "show %s models; " % selected[0].structure.atomspec
            from chimerax.atomic import concise_residue_spec
            spec = concise_residue_spec(self.session, selected)
            cmd = base_cmd + f"select {spec}; view {spec}"
        from chimerax.core.commands import run
        run(self.session, cmd)

    def _update_button_texts(self):
        all_input, douse_only, douse_in_common, input_in_common = self.compared_waters
        self.douse_only_button.setText("Douse only (%d)" % len(douse_only))
        self.in_common_button.setText("In common (%d)" % len(douse_in_common))
        self.original_only_button.setText("Input only (%d)" % len(all_input - input_in_common))

    def _update_residues(self):
        all_input, douse_only, douse_in_common, input_in_common = self.compared_waters
        checked = self.radio_group.checkedButton()
        if checked == self.douse_only_button:
            self.filter_model = self.douse_model
            self.filter_residues = douse_only
        elif checked == self.original_only_button:
            self.filter_model = self.orig_model
            self.filter_residues = all_input - input_in_common
        else:
            self.filter_model = self.douse_model
            self.filter_residues = douse_in_common
        self.res_list.refresh()
