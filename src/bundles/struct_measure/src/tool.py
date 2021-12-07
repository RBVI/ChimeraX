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

_tool = None
def get_tool(session, tool_name):
    global _tool
    if _tool is None:
        _tool = StructMeasureTool(session)
    _tool.show_tab(tool_name)
    return _tool

from chimerax.core.tools import ToolInstance
from Qt.QtWidgets import QTableWidget, QHBoxLayout, QVBoxLayout, QAbstractItemView, QWidget, QPushButton, \
    QTabWidget, QTableWidgetItem, QFileDialog

class StructMeasureTool(ToolInstance):

    tab_names = ["Distances", "Angles/Torsions"]
    help_info = ["distances", "angles"]

    def __init__(self, session):
        ToolInstance.__init__(self, session, "Structure Measurements")
        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area
        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self._set_help)
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        parent.setLayout(layout)
        layout.addWidget(self.tab_widget)
        self.handlers = []
        for tab_name in self.tab_names:
            self._add_tab(tab_name)

        tw.manage(placement="side")

    def delete(self):
        global _tool
        _tool = None
        for handler in self.handlers:
            handler.remove()
        super().delete()

    def show_tab(self, tab_name):
        index = self.tab_names.index(tab_name)
        self.tab_widget.setCurrentIndex(index)

    def _add_tab(self, tab_name):
        tab_area = QWidget()
        if tab_name == "Distances":
            self._fill_distance_tab(tab_area)
        elif tab_name == "Angles/Torsions":
            self._fill_angle_tab(tab_area)
        else:
            raise AssertionError("Don't know how to create structure-measurement tab '%s'" % tab_name)
        self.tab_widget.addTab(tab_area, tab_name)

    def _angle_changes_handler(self, _, changes):
        if changes.num_deleted_atoms() > 0:
            self._update_angles()
        elif "position changed" in changes.structure_reasons() \
        or len(changes.modified_coordsets()) > 0:
            self._update_angles()
        if "active_coordset changed" in changes.structure_reasons():
            self._update_angles()

    def _angle_text(self, atoms):
        from chimerax.geometry import angle, dihedral
        func = angle if len(atoms) == 3 else dihedral
        return self._angle_fmt % func(*[a.scene_coord for a in atoms])

    def _create_angle(self):
        from chimerax.atomic import selected_atoms
        sel_atoms = selected_atoms(self.session)
        if len(sel_atoms) not in (3, 4):
            from chimerax.core.errors import UserError
            raise UserError("Either three or four atoms must be selected!")
        num_rows = self.angle_table.rowCount()
        self.angle_table.insertRow(num_rows)
        strings = [sel_atoms[0].string()]
        prev = sel_atoms[0]
        for a in sel_atoms[1:]:
            strings.append(a.string(relative_to=prev))
            prev = a
        for col, string in enumerate(strings):
            self.angle_table.setItem(num_rows, col, QTableWidgetItem(string))
        self.angle_table.setItem(num_rows, 4, QTableWidgetItem(self._angle_text(sel_atoms)))
        for i in range(self.angle_table.columnCount()):
            self.angle_table.resizeColumnToContents(i)
        self._angle_info.append((sel_atoms, len(sel_atoms)))

    def _create_distance(self):
        from chimerax.atomic import selected_atoms
        sel_atoms = selected_atoms(self.session)
        if len(sel_atoms) != 2:
            from chimerax.core.errors import UserError
            raise UserError("Exactly two atoms must be selected!")
        from chimerax.core.commands import run
        run(self.session, "distance %s %s" % tuple(a.string(style="command") for a in sel_atoms))

    def _delete_angle(self):
        from chimerax.core.errors import UserError
        rows = set([index.row() for index in self.angle_table.selectedIndexes()])
        if not rows:
            raise UserError("Must select one or more angles/torsions in the table")
        for row in reversed(sorted(list(rows))):
            self.angle_table.removeRow(row)
            del self._angle_info[row]

    def _delete_distance(self):
        from chimerax.core.errors import UserError
        from chimerax.core.commands import run
        dist_grp = self.session.pb_manager.get_group("distances", create=False)
        if not dist_grp:
            raise UserError("No distances to delete!")
        pbs = dist_grp.pseudobonds
        if not pbs:
            raise UserError("No distances to delete!")
        rows = set([index.row() for index in self.dist_table.selectedIndexes()])
        if not rows:
            raise UserError("Must select one or more distances in the table")
        del_pbs = []
        for i, pb in enumerate(pbs):
            if i in rows:
                del_pbs.append(pb)
        for pb in del_pbs:
            run(self.session, "~distance %s %s" % tuple([a.string(style="command") for a in pb.atoms]))

    def _fill_angle_tab(self, tab_area):
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        tab_area.setLayout(layout)
        self.angle_table = QTableWidget()
        self.angle_table.setColumnCount(5)
        self.angle_table.keyPressEvent = self.session.ui.forward_keystroke
        self.angle_table.setHorizontalHeaderLabels(["Atom 1", "Atom 2", "Atom 3", "Atom 4", "Angle/Torsion"])
        self.angle_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.angle_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.angle_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table_layout = QVBoxLayout()
        table_layout.setContentsMargins(0,0,0,0)
        table_layout.setSpacing(0)
        table_layout.addWidget(self.angle_table)
        table_layout.setStretchFactor(self.angle_table, 1)
        layout.addLayout(table_layout)
        layout.setStretchFactor(table_layout, 1)
        button_layout = QHBoxLayout()
        create_button = QPushButton("Create")
        create_button.clicked.connect(self._create_angle)
        create_button.setToolTip("Show angle/torsion value between three/four (currently selected) atoms")
        button_layout.addWidget(create_button)
        delete_button = QPushButton("Delete")
        delete_button.clicked.connect(self._delete_angle)
        delete_button.setToolTip("Delete angles/torsions selected in table (or all if none selected)")
        button_layout.addWidget(delete_button)
        save_info_button = QPushButton("Save Info...")
        save_info_button.clicked.connect(self._save_angle_info)
        save_info_button.setToolTip("Save angle/torsion information into a file")
        button_layout.addWidget(save_info_button)
        table_layout.addLayout(button_layout)
        self._angle_info = []

        from .settings import get_settings
        settings = get_settings(self.session, "angles")
        from chimerax.ui.options import SettingsPanel, IntOption
        panel = SettingsPanel()
        for opt_name, attr_name, opt_class, opt_class_kw in [
                ("Decimal places", 'decimal_places', IntOption, {'min': 0})]:
            panel.add_option(opt_class(opt_name, None,
                lambda opt, settings=settings: self._set_angle_decimal_places(settings.decimal_places),
                attr_name=attr_name, settings=settings, auto_set_attr=True))
        layout.addWidget(panel)
        self._set_angle_decimal_places(settings.decimal_places)

        from chimerax.atomic import get_triggers
        self.handlers.append(get_triggers().add_handler('changes', self. _angle_changes_handler))

    def _fill_dist_table(self, *args):
        dist_grp = self.session.pb_manager.get_group("distances", create=False)
        if not dist_grp:
            self.dist_table.clearContents()
            self.dist_table.setRowCount(0)
            return
        fmt = self.session.pb_dist_monitor.distance_format
        pbs = dist_grp.pseudobonds
        update = len(pbs) == self.dist_table.rowCount()
        if not update:
            self.dist_table.clearContents()
            self.dist_table.setRowCount(len(pbs))
        for row, pb in enumerate(pbs):
            a1, a2 = pb.atoms
            strings = a1.string(), a2.string(relative_to=a1), fmt % pb.length
            for col, string in enumerate(strings):
                if update:
                    self.dist_table.item(row, col).setText(string)
                else:
                    self.dist_table.setItem(row, col, QTableWidgetItem(string))
        for i in range(self.dist_table.columnCount()):
            self.dist_table.resizeColumnToContents(i)

    def _fill_distance_tab(self, tab_area):
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        tab_area.setLayout(layout)
        self.dist_table = QTableWidget()
        self.dist_table.setColumnCount(3)
        self.dist_table.keyPressEvent = self.session.ui.forward_keystroke
        self.dist_table.setHorizontalHeaderLabels(["Atom 1", "Atom 2", "Distance"])
        self.dist_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.dist_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.dist_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table_layout = QVBoxLayout()
        table_layout.setContentsMargins(0,0,0,0)
        table_layout.setSpacing(0)
        table_layout.addWidget(self.dist_table)
        table_layout.setStretchFactor(self.dist_table, 1)
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
        save_info_button.clicked.connect(self._save_dist_info)
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
        self.handlers.extend([
            group_triggers.add_handler("update", self._fill_dist_table),
            group_triggers.add_handler("delete", self._fill_dist_table)
        ])
        self._fill_dist_table()

    def _save_angle_info(self):
        from chimerax.core.errors import UserError
        if not self._angle_info:
            raise UserError("No angles/torsions to save!")
        path = QFileDialog.getSaveFileName(self.angle_table, "Save Angles/Torsions File")[0]
        with open(path, 'wt') as f:
            for atoms, num_expected_atoms in self._angle_info:
                atom_strings = [str(atoms[0])]
                prev = atoms[0]
                for a in atoms[1:]:
                    atom_strings.append(a.string(relative_to=prev))
                    prev = a
                print("%s: %s" % (', '.join(atom_strings), self._angle_text(atoms)), file=f)

    def _save_dist_info(self):
        from chimerax.core.errors import UserError
        from chimerax.core.commands import run
        dist_grp = self.session.pb_manager.get_group("distances", create=False)
        if not dist_grp:
            raise UserError("No distances to save!")
        pbs = dist_grp.pseudobonds
        if not pbs:
            raise UserError("No distances to save!")
        run(self.session, "distance save browse")

    def _set_angle_decimal_places(self, decimal_places):
        self._angle_fmt = "%%.%df\N{DEGREE SIGN}" % decimal_places
        self._update_angles()

    def _set_help(self, index):
        help_text = self.help_info[index]
        if help_text:
            self.help = "help:user/tools/%s.html" % help_text
        else:
            self.help = None

    def _update_angles(self):
        next_angle_info = []
        death_row = []
        for i, angle_info in enumerate(self._angle_info):
            atoms, expected_num_atoms = angle_info
            if len(atoms) != expected_num_atoms:
                death_row.append(i)
                continue
            next_angle_info.append(angle_info)
            self.angle_table.item(i, 4).setText(self._angle_text(atoms))
        self._angle_info = next_angle_info
        for row in reversed(death_row):
            self.angle_table.removeRow(row)
