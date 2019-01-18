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
class ModellerLauncher(ToolInstance):
    """Generate the inputs needed by Modeller for comparitive modeling"""

    #help = "help:user/tools/sequenceviewer.html"
    SESSION_SAVE = False

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        self.tool_window = MainToolWindow(self, close_destroys=False, statusbar=False)
        parent = self.tool_window.ui_area

        from PyQt5.QtWidgets import QListWidget, QGridLayout, QAbstractItemView, QWidget, QVBoxLayout
        alignments_layout = QVBoxLayout()
        alignments_layout.setContentsMargins(0,0,0,0)
        alignments_layout.setSpacing(0)
        parent.setLayout(alignments_layout)
        self.alignment_list = QListWidget()
        self.alignment_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.alignment_list.keyPressEvent = session.ui.forward_keystroke
        self.alignment_list.itemSelectionChanged.connect(self._list_selection_cb)
        alignments_layout.addWidget(self.alignment_list)
        alignments_layout.setStretchFactor(self.alignment_list, 1)
        targets_area = QWidget()
        # would use a QFormLayout, but can't easily hide rows...
        self.targets_layout = QGridLayout()
        self.targets_layout.setColumnStretch(1, 1)
        targets_area.setLayout(self.targets_layout)
        alignments_layout.addWidget(targets_area)
        self.alignments = []
        self.target_sequences = {}
        self.target_menus = {}
        self.target_labels = {}
        self._refresh_alignments()
        self.tool_window.manage('side')

    def delete(self):
        ToolInstance.delete(self)

    def _make_target_menu(self, alignment):
        no_target_text = "No target sequence selected"
        from PyQt5.QtWidgets import QPushButton, QMenu, QLabel
        self.target_menus[alignment] = menu_button = QPushButton(no_target_text)
        menu = QMenu()
        menu_button.setMenu(menu)
        menu_button.hide()
        menu.addAction(no_target_text)
        for seq in alignment.seqs:
            menu.addAction(seq.name)
        def menu_action(act, mb=menu_button, aln=alignment):
            menu_index = mb.menu().actions().index(act)
            if menu_index == 0:
                self.target_sequences[aln] = None
            else:
                self.target_sequences[aln] = aln.seqs[menu_index-1]
            mb.setText(act.text())
        menu.triggered.connect(menu_action)
        self.target_labels[alignment] = label = QLabel(alignment.ident)
        label.hide()
        row = self.targets_layout.rowCount()
        from PyQt5.QtCore import Qt
        self.targets_layout.addWidget(label, row, 0, Qt.AlignRight)
        self.targets_layout.addWidget(menu_button, row, 1)

    def _refresh_alignments(self):
        self.alignment_list.blockSignals(True)
        current_alignments = self.session.alignments.alignments
        row_order_alignments = []
        for old_alignment in self.alignments:
            if old_alignment not in current_alignments:
                self.target_labels[old_alignment].destroy()
                self.target_menus[old_alignment].destroy()
                del self.target_labels[old_alignment]
                del self.target_menus[old_alignment]
                del self.target_sequences[old_alignment]
            else:
                row_order_alignments.append(old_alignment)
        for cur_alignment in current_alignments:
            if cur_alignment not in self.alignments:
                self._make_target_menu(cur_alignment)
                self.target_sequences[cur_alignment] = None
                row_order_alignments.append(cur_alignment)
        self.alignments = row_order_alignments
        self.alignment_list.clear()
        self.alignment_list.addItems([str(aln) for aln in self.alignments])
        from PyQt5.QtCore import QItemSelectionModel
        for row, aln in enumerate(self.alignments):
            if not self.target_menus[aln].isHidden():
                self.alignment_list.setCurrentRow(row, QItemSelectionModel.SelectCurrent)
        self.alignment_list.blockSignals(False)

    def _list_selection_cb(self, *args):
        sel_rows = set([i.row() for i in self.alignment_list.selectedIndexes()])
        for row, aln in enumerate(self.alignments):
            hidden = row not in sel_rows
            self.target_labels[aln].setHidden(hidden)
            self.target_menus[aln].setHidden(hidden)

class ModellerResultsViewer(ToolInstance):
    """ Viewer displays the models/results generated by Modeller"""

    #help = "help:user/tools/sequenceviewer.html"

    def __init__(self, session, tool_name, models=None, attr_names=None):
        """ if 'models' is None, then we are being restored from a session and
            set_state_from_snapshot will be called later.
        """

        ToolInstance.__init__(self, session, tool_name)
        if models is None:
            return
        self._finalize_init(session, models, attr_names)

    def _finalize_init(self, session, models, attr_names):
        self.models = models
        self.attr_names = attr_names
        from chimerax.core.models import REMOVE_MODELS
        self.model_handler = session.triggers.add_handler(REMOVE_MODELS, self._models_removed_cb)

        from chimerax.ui import MainToolWindow
        self.tool_window = MainToolWindow(self, close_destroys=False, statusbar=False)
        #TODO: self.tool_window.fill_context_menu = self.fill_context_menu
        parent = self.tool_window.ui_area

        from PyQt5.QtWidgets import QTableWidget, QVBoxLayout, QAbstractItemView, QWidget, QPushButton
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        parent.setLayout(layout)
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setSortingEnabled(True)
        self.table.keyPressEvent = session.ui.forward_keystroke
        self.table.setHorizontalHeaderLabels(["Model"] + [attr_name[9:] for attr_name in attr_names])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.itemSelectionChanged.connect(self._table_selection_cb)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.table)
        layout.setStretchFactor(self.table, 1)
        self._fill_table()
        self.tool_window.manage('side')

    def delete(self):
        self.model_handler.remove()
        self.row_item_lookup = {}
        ToolInstance.delete(self)

    #TODO
    def fill_context_menu(self, menu, x, y):
        # avoid having actions destroyed when this routine returns
        # by stowing a reference in the menu itself
        from PyQt5.QtWidgets import QAction
        save_as_menu = menu.addMenu("Save As")
        from chimerax.core import io
        from chimerax.core.commands import run, quote_if_necessary
        for fmt in io.formats(open=False):
            if fmt.category == "Sequence alignment":
                action = QAction(fmt.name, save_as_menu)
                action.triggered.connect(lambda arg, fmt=fmt:
                    run(self.session, "save browse format %s alignment %s"
                    % (fmt.name, quote_if_necessary(self.alignment.ident))))
                save_as_menu.addAction(action)

        settings_action = QAction("Settings...", menu)
        settings_action.triggered.connect(lambda arg: self.show_settings())
        menu.addAction(settings_action)
        scf_action = QAction("Load Sequence Coloring File...", menu)
        scf_action.triggered.connect(lambda arg: self.load_scf_file(None))
        menu.addAction(scf_action)

    @classmethod
    def restore_snapshot(cls, session, data):
        inst = super().restore_snapshot(session, data['ToolInstance'])
        inst._finalize_init(session, data['models'], data['attr_names'])
        return inst

    SESSION_SAVE = True

    def take_snapshot(self, session, flags):
        data = {
            'ToolInstance': ToolInstance.take_snapshot(self, session, flags),
            'models': self.models,
            'attr_names': self.attr_names
        }
        return data

    def _fill_table(self):
        self.table.clearContents()
        self.table.setRowCount(len(self.models))
        from PyQt5.QtWidgets import QTableWidgetItem
        for row, m in enumerate(self.models):
            item = QTableWidgetItem('#' + m.id_string)
            self.table.setItem(row, 0, item)
            for c, attr_name in enumerate(self.attr_names):
                self.table.setItem(row, c+1, QTableWidgetItem("%g" % getattr(m, attr_name, "")))
        for i in range(self.table.columnCount()):
            self.table.resizeColumnToContents(i)

    def _models_removed_cb(self, *args):
        remaining = [m for m in self.models if m.id is not None]
        if remaining == self.models:
            return
        self.models = remaining
        if not self.models:
            self.delete()
        else:
            self._fill_table()

    def _table_selection_cb(self):
        rows = set([index.row() for index in self.table.selectedIndexes()])
        sel_ids = set([self.table.item(r, 0).text() for r in rows])
        for m in self.models:
            m.display = ('#' + m.id_string) in sel_ids or not sel_ids
