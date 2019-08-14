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

class SelInspector(ToolInstance):

    #help = "help:user/tools/distances.html"

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)
        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area
        from PyQt5.QtWidgets import QVBoxLayout, QWidget, QLabel, QPushButton, QMenu
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        parent.setLayout(layout)

        self.text_description = QLabel()
        layout.addWidget(self.text_description)

        self.chooser = QPushButton()
        self.item_menu = QMenu()
        self.chooser.setMenu(self.item_menu)
        layout.addWidget(self.chooser)

        self.options_area = QWidget()
        layout.addWidget(self.options_area)

        #TODO
        from chimerax.dist_monitor.cmd import group_triggers
        self.handlers = [
            group_triggers.add_handler("update", self._fill_table),
            group_triggers.add_handler("delete", self._fill_table)
        ]
        self._fill_table()
        tw.manage(placement="side")

    def delete(self):
        for handler in self.handlers:
            handler.remove()
        super().delete()

    def _create_distance(self):
        from chimerax.atomic import selected_atoms
        sel_atoms = selected_atoms(self.session)
        if len(sel_atoms) != 2:
            from chimerax.core.errors import UserError
            raise UserError("Exactly two atoms must be selected!")
        from chimerax.core.commands import run
        run(self.session, "distance %s %s" % tuple(a.string(style="command") for a in sel_atoms))

    def _delete_distance(self):
        from chimerax.core.errors import UserError
        from chimerax.core.commands import run
        dist_grp = self.session.pb_manager.get_group("distances", create=False)
        if not dist_grp:
            raise UserError("No distances to delete!")
        pbs = dist_grp.pseudobonds
        if not pbs:
            raise UserError("No distances to delete!")
        rows = set([index.row() for index in self.table.selectedIndexes()])
        if not rows:
            raise UserError("Must select one or more distances in the table")
        del_pbs = []
        for i, pb in enumerate(pbs):
            if i in rows:
                del_pbs.append(pb)
        for pb in del_pbs:
            run(self.session, "~distance %s %s" % tuple([a.string(style="command") for a in pb.atoms]))

    def _save_info(self):
        from chimerax.core.errors import UserError
        from chimerax.core.commands import run
        dist_grp = self.session.pb_manager.get_group("distances", create=False)
        if not dist_grp:
            raise UserError("No distances to save!")
        pbs = dist_grp.pseudobonds
        if not pbs:
            raise UserError("No distances to save!")
        run(self.session, "distance save browse")

    def _fill_table(self, *args):
        dist_grp = self.session.pb_manager.get_group("distances", create=False)
        if not dist_grp:
            self.table.clearContents()
            self.table.setRowCount(0)
            return
        fmt = self.session.pb_dist_monitor.distance_format
        from PyQt5.QtWidgets import QTableWidgetItem
        pbs = dist_grp.pseudobonds
        update = len(pbs) == self.table.rowCount()
        if not update:
            self.table.clearContents()
            self.table.setRowCount(len(pbs))
        for row, pb in enumerate(pbs):
            a1, a2 = pb.atoms
            strings = a1.string(), a2.string(relative_to=a1), fmt % pb.length
            for col, string in enumerate(strings):
                if update:
                    self.table.item(row, col).setText(string)
                else:
                    self.table.setItem(row, col, QTableWidgetItem(string))
        for i in range(self.table.columnCount()):
            self.table.resizeColumnToContents(i)

