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


class DistanceTool(ToolInstance):

    help = "help:user/tools/distances.html"

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)
        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area
        from Qt.QtWidgets import QTableWidget, QHBoxLayout, QVBoxLayout, QAbstractItemView, QWidget, \
            QPushButton
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        parent.setLayout(layout)
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.keyPressEvent = session.ui.forward_keystroke
        self.table.setHorizontalHeaderLabels(["Atom 1", "Atom 2", "Distance"])
        #self.table.itemClicked.connect(self._table_change_cb)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table_layout = QVBoxLayout()
        table_layout.setContentsMargins(0,0,0,0)
        table_layout.setSpacing(0)
        table_layout.addWidget(self.table)
        table_layout.setStretchFactor(self.table, 1)
        layout.addLayout(table_layout)
        layout.setStretchFactor(table_layout, 1)
        button_layout = QHBoxLayout()
        create_button = QPushButton("Create")
        create_button.clicked.connect(self._create_distance)
        create_button.setToolTip("Create distance monitor between two (currently selected) atoms;\n"
            "Alternatively, control-click one atom in graphics view and control-shift-\n"
            "double-click another to bring up context menu with 'Distance' entry")
        button_layout.addWidget(create_button)
        delete_button = QPushButton("Delete")
        delete_button.clicked.connect(self._delete_distance)
        delete_button.setToolTip("Delete distances selected in table (or all if none selected)")
        button_layout.addWidget(delete_button)
        save_info_button = QPushButton("Save Info...")
        save_info_button.clicked.connect(self._save_info)
        save_info_button.setToolTip("Save distance information into a file")
        button_layout.addWidget(save_info_button)
        table_layout.addLayout(button_layout)

        from chimerax.ui.options import SettingsPanel, BooleanOption, ColorOption, IntOption, FloatOption
        panel = SettingsPanel()
        from chimerax.dist_monitor.settings import settings
        from chimerax.core.commands import run
        from chimerax.core.colors import color_name
        for opt_name, attr_name, opt_class, opt_class_kw, cmd_arg in [
                ("Color", 'color', ColorOption, {}, 'color %s'),
                ("Number of dashes", 'dashes', IntOption, {'min': 0}, 'dashes %d'),
                ("Decimal places", 'decimal_places', IntOption, {'min': 0}, 'decimalPlaces %d'),
                ("Radius", 'radius', FloatOption, {'min': 'positive', 'decimal_places': 3}, 'radius %g'),
                ("Show \N{ANGSTROM SIGN} symbol", 'show_units', BooleanOption, {}, 'symbol %s')]:
            converter = color_name if opt_class == ColorOption else None
            panel.add_option(opt_class(opt_name, None,
                lambda opt, run=run, converter=converter, ses=self.session, cmd_suffix=cmd_arg:
                run(ses, "distance style " + cmd_suffix
                % (opt.value if converter is None else converter(opt.value))),
                attr_name=attr_name, settings=settings, auto_set_attr=False))
        layout.addWidget(panel)

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
        from Qt.QtWidgets import QTableWidgetItem
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

