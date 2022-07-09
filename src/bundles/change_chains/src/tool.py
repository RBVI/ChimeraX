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

_ccid = None
def change_chain_ids_dialog(session, tool_name):
    global _ccid
    if _ccid is None:
        _ccid = ChangeChainIDsDialog(session, tool_name)
    return _ccid

class ChangeChainIDsDialog(ToolInstance):

    #help = "help:user/tools/renumber.html"
    SESSION_SAVE = False

    single_text = "To one ID"
    multiple_text = "To multiple IDs"

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)
        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area
        from Qt.QtWidgets import QHBoxLayout, QVBoxLayout, QTabWidget, QWidget, QListWidget, QLabel
        from Qt.QtWidgets import QCheckBox, QGridLayout
        from Qt.QtCore import Qt
        self.layout = layout = QVBoxLayout()
        parent.setLayout(layout)
        layout.setContentsMargins(10,0,10,0)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, stretch=1)

        from chimerax.ui.options import OptionsPanel, StringOption
        one_id_widget = OptionsPanel()
        self.single_id = StringOption("Change residues chain ID to", "", None)
        self.single_id.widget.setMaximumWidth(5 * self.single_id.widget.fontMetrics().averageCharWidth())
        one_id_widget.add_option(self.single_id)
        self.tabs.addTab(one_id_widget, self.single_text)

        multi_widget = QWidget()
        multi_layout = QHBoxLayout()
        multi_widget.setLayout(multi_layout)
        self.chain_list = QListWidget()
        self.chain_list.setSelectionMode(self.chain_list.ExtendedSelection)
        self.chain_list.setMaximumWidth(5 * self.chain_list.fontMetrics().averageCharWidth())
        multi_layout.addWidget(self.chain_list)
        self.widget_mapping = {}
        self.unused_widgets = []
        self.widgets_layout = QGridLayout()
        self.widgets_layout.setSpacing(0)
        multi_layout.addLayout(self.widgets_layout)
        self.chain_list.itemSelectionChanged.connect(self._update_chain_list)
        self.tip = QLabel("Choose one or more chain IDs from the list on the left.  "
            "Doing so will cause corresponding entry fields to appear here, where"
            " you can enter the chain IDs to change to.")
        self.tip.setWordWrap(True)
        self.widgets_layout.addWidget(self.tip, 0, 0)
        self._update_chain_list()
        from chimerax.atomic import get_triggers
        self.handlers = [
            get_triggers().add_handler("changes", self._possibly_update_chains),
        ]
        self.tabs.addTab(multi_widget, self.multiple_text)

        self.sel_restrict = QCheckBox("Restrict change to selected residues, if any")
        self.sel_restrict.setChecked(True)
        layout.addWidget(self.sel_restrict, alignment=Qt.AlignCenter)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        self.bbox = bbox = qbbox(qbbox.Ok | qbbox.Apply | qbbox.Close | qbbox.Help)
        bbox.accepted.connect(self.change_chain_ids)
        bbox.button(qbbox.Apply).clicked.connect(self.change_chain_ids)
        bbox.accepted.connect(self.delete) # slots executed in the order they are connected
        bbox.rejected.connect(self.delete)
        from chimerax.core.commands import run
        #bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help " + self.help))
        bbox.button(qbbox.Help).setEnabled(False)
        layout.addWidget(bbox)

        tw.manage(placement=None)

    def delete(self):
        global _ccid
        _ccid = None
        for handler in self.handlers:
            handler.remove()
        super().delete()

    def change_chain_ids(self):
        cmd = "changechains "
        spec_present = False
        if self.sel_restrict.isChecked():
            from chimerax.atomic import selected_residues
            sel_res = selected_residues(self.session)
            if sel_res:
                cmd += "sel "
                spec_present = True
        if self.tabs.tabText(self.tabs.currentIndex()) == self.single_text:
            cid = self.single_id.value
            if not cid:
                raise UserError("Cannot change to empty ID")
            cmd += cid
        else:
            from_ids = []
            to_ids = []
            for from_id, widgets in self.widget_mapping.items():
                label, entry = widgets
                if label.isHidden():
                    continue
                from_ids.append(from_id)
                to_id = entry.text()
                if not to_id:
                    raise UserError("Cannot change to an empty ID")
                to_ids.append(to_id)
            from chimerax.core.commands import StringArg
            from_list = ','.join([StringArg.unparse(cid) for cid in from_ids])
            # the atom spec parser coughs up a hair ball for some quoted strings, so
            # prevent the atom spec parser from seeing such a thing
            if not from_list[0].isalnum() and not spec_present:
                cmd += "#* "
            cmd += from_list
            cmd += ' '
            cmd += ','.join([StringArg.unparse(cid) for cid in to_ids])

        from chimerax.core.commands import run
        run(self.session, cmd)

    def _possibly_update_chains(self, trig_name, trig_data):
        if trig_data.created_residues() or trig_data.num_deleted_residues() > 0 \
        or "chain_id changed" in trig_data.residue_reasons():
            self._update_chain_list()

    def _update_chain_list(self):
        self.chain_list.blockSignals(True)
        if not self.tip.isHidden():
            self.tip.setHidden(True)
            self.widgets_layout.removeWidget(self.tip)
        from chimerax.atomic import all_structures
        chain_ids = sorted(all_structures(self.session).residues.unique_chain_ids)
        chain_id_set = set(chain_ids)
        sel_chain_ids = set(item.text() for item in self.chain_list.selectedItems())
        for cid, widgets in list(self.widget_mapping.items())[:]:
            for widget in widgets:
                widget.setHidden(True)
                self.widgets_layout.removeWidget(widget)
            if cid not in chain_id_set:
                del self.widget_mapping[cid]
                self.unused_widgets.append(widgets)
        from Qt.QtCore import Qt, QItemSelectionModel
        for index, cid in enumerate(chain_ids):
            try:
                label, entry = widgets = self.widget_mapping[cid]
            except KeyError:
                text = cid + "\N{RIGHTWARDS ARROW}"
                if self.unused_widgets:
                    label, entry = self.unused_widgets.pop()
                    label.setText(text)
                else:
                    from Qt.QtWidgets import QLabel, QLineEdit
                    label = QLabel(text)
                    entry = QLineEdit()
                    entry.setMaximumWidth(5 * entry.fontMetrics().averageCharWidth())
                widgets = (label, entry)
                self.widget_mapping[cid] = widgets
            self.widgets_layout.addWidget(label, index, 0, alignment=Qt.AlignRight)
            self.widgets_layout.addWidget(entry, index, 1, alignment=Qt.AlignLeft)
            for widget in widgets:
                widget.setHidden(cid not in sel_chain_ids)
        self.chain_list.clear()
        self.chain_list.addItems(chain_ids)
        some_selected = False
        for row in range(self.chain_list.count()):
            item = self.chain_list.item(row)
            if item.text() in sel_chain_ids:
                self.chain_list.setCurrentItem(item, QItemSelectionModel.Select)
                some_selected = True
        if not some_selected:
            self.tip.setHidden(False)
            self.widgets_layout.addWidget(self.tip, 0, 0)
        self.chain_list.blockSignals(False)

