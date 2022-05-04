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
    def __init__(self, session, tool_name, orig_model=None, douse_model=None, map=None, map_range=None,
            compared_waters=None):
        # if 'model' is None, we are being restored from a session
        # and set_state_from_snapshot will be called later
        super().__init__(session, tool_name)
        if douse_model is None:
            return
        self._finalize_init(orig_model, douse_model, map, map_range, compared_waters)

    def _finalize_init(self, orig_model, douse_model, map, map_range, compared_waters):
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
            but_layout = QVBoxLayout()
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
        else:
            # didn't keep the input waters
            #TODO
            self.radio_group = None
        self.filter_model = self.douse_model
        self.filter_residues = compared_waters[1]
        from chimerax.atomic.widgets import ResidueListWidget
        self.res_list = ResidueListWidget(self.session, filter_func=self._filter_residues)
        layout.addWidget(self.res_list)

        self.tool_window.manage('side')

    def delete(self):
        for handler in self.handlers:
            handler.remove()
        self.orig_model = self.douse_model = None
        super().delete()

    def _filter_residues(self, r):
        return r.structure == self.filter_model and r in self.filter_residues

    def _models_removed_cb(self, *args):
        if self.orig_model.id is None or self.douse_model.id is None:
            self.delete()

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
