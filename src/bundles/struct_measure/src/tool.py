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

from chimerax.core.errors import UserError
from chimerax.core.tools import ToolInstance
from Qt.QtWidgets import QTableWidget, QHBoxLayout, QVBoxLayout, QAbstractItemView, QWidget, QPushButton, \
    QTabWidget, QTableWidgetItem, QFileDialog, QDialogButtonBox as qbbox, QLabel, QButtonGroup, \
    QRadioButton, QLineEdit, QGroupBox, QGridLayout
from Qt.QtGui import QDoubleValidator
from Qt.QtCore import Qt
from chimerax.ui.widgets import ColorButton

class StructMeasureTool(ToolInstance):

    tab_names = ["Distances", "Angles/Torsions", "Axes/Planes/Centroids"]
    help_info = ["distances", "angles", None]

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
        elif tab_name == "Axes/Planes/Centroids":
            self._fill_axis_tab(tab_area)
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
            raise UserError("Exactly two atoms must be selected!")
        from chimerax.core.commands import run
        run(self.session, "distance %s %s" % tuple(a.string(style="command") for a in sel_atoms))

    def _delete_angle(self):
        rows = set([index.row() for index in self.angle_table.selectedIndexes()])
        if not rows:
            raise UserError("Must select one or more angles/torsions in the table")
        for row in reversed(sorted(list(rows))):
            self.angle_table.removeRow(row)
            del self._angle_info[row]

    def _delete_distance(self):
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
        from chimerax.core.commands import run
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

    def _fill_axis_tab(self, tab_area):
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        tab_area.setLayout(layout)

        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)
        axis_button = QPushButton("Define axes...")
        axis_button.clicked.connect(lambda *args, cb=self._show_define_axis_dialog: cb())
        button_layout.addWidget(axis_button, alignment=Qt.AlignHCenter)
        plane_button = QPushButton("Define planes...")
        plane_button.clicked.connect(lambda *args, cb=self._show_define_plane_dialog: cb())
        button_layout.addWidget(plane_button, alignment=Qt.AlignHCenter)
        centroid_button = QPushButton("Define centroids...")
        centroid_button.clicked.connect(lambda *args, cb=self._show_define_centroid_dialog: cb())
        button_layout.addWidget(centroid_button, alignment=Qt.AlignHCenter)
        self._define_axis_dialog = self._define_plane_dialog = self._define_centroid_dialog = None

        """
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
        """

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
        dist_grp = self.session.pb_manager.get_group("distances", create=False)
        if not dist_grp:
            raise UserError("No distances to save!")
        pbs = dist_grp.pseudobonds
        if not pbs:
            raise UserError("No distances to save!")
        from chimerax.core.commands import run
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

    def _show_define_axis_dialog(self):
        if not self._define_axis_dialog:
            self._define_axis_dialog = DefineAxisDialog(self)
        self._define_axis_dialog.tool_window.shown = True

    def _show_define_centroid_dialog(self):
        pass

    def _show_define_plane_dialog(self):
        pass

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

class DefineAxisDialog:
    def __init__(self, sm_tool):
        self.tool_window = tw = sm_tool.tool_window.create_child_window("Define Axes", close_destroys=False)
        self.session = sm_tool.session
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        tw.ui_area.setLayout(layout)

        controls_layout = QHBoxLayout()
        layout.addLayout(controls_layout)

        type_layout = QVBoxLayout()
        controls_layout.addLayout(type_layout)
        type_layout.addWidget(QLabel("Create axis for..."), alignment=Qt.AlignLeft | Qt.AlignTop)
        self.button_group = QButtonGroup()
        self.button_group.buttonClicked.connect(self.show_applicable_params)
        self.shown_for_button = {} # widgets that are always shown don't go into this
        self.button_dispatch = {}
        self.axis_name_for_button = {}

        helix_button = QRadioButton("Each helix in structure(s)")
        helix_button.setChecked(True)
        self.button_group.addButton(helix_button)
        type_layout.addWidget(helix_button, alignment=Qt.AlignLeft | Qt.AlignTop)
        self.shown_for_button[helix_button] = set()
        self.axis_name_for_button[helix_button] = "helix axes"
        self.button_dispatch[helix_button] = self.cmd_params_helix_axis

        """
        atoms_layout = QHBoxLayout()
        type_layout.addLayout(atoms_layout)
        self.atoms_button = QRadioButton("Selected atoms/centroids (axis name: ")
        self.atoms_button.setChecked(False)
        self.button_group.addButton(self.atoms_button)
        atoms_layout.addWidget(self.atoms_button)
        self.atoms_name_edit = QLineEdit()
        self.atoms_name_edit.setText("axis")
        atoms_layout.addWidget(self.atoms_name_edit)
        atoms_layout.setStretch(atoms_layout.count()-1, 1)
        atoms_layout.addWidget(QLabel(")"))
        """
        atoms_button = QRadioButton("Selected atoms/centroids")
        atoms_button.setChecked(False)
        self.button_group.addButton(atoms_button)
        type_layout.addWidget(atoms_button, alignment=Qt.AlignLeft | Qt.AlignTop)
        self.shown_for_button[atoms_button] = set()
        self.axis_name_for_button[atoms_button] = "axis"
        self.button_dispatch[atoms_button] = self.cmd_params_atoms_axis

        plane_button = QRadioButton("Plane normal(s)")
        plane_button.setChecked(False)
        self.button_group.addButton(plane_button)
        type_layout.addWidget(plane_button, alignment=Qt.AlignLeft | Qt.AlignTop)
        self.shown_for_button[plane_button] = set()
        self.axis_name_for_button[plane_button] = "normal"
        self.button_dispatch[plane_button] = self.cmd_params_plane_axis

        points_button = QRadioButton("Two points")
        points_button.setChecked(False)
        self.button_group.addButton(points_button)
        type_layout.addWidget(points_button, alignment=Qt.AlignLeft | Qt.AlignTop)
        self.shown_for_button[points_button] = set()
        self.axis_name_for_button[points_button] = "axis"
        self.button_dispatch[points_button] = self.cmd_params_points_axis

        params_layout = QVBoxLayout()
        controls_layout.addLayout(params_layout)
        self.all_params_widgets = []

        structure_label = QLabel("Structure(s)")
        params_layout.addWidget(structure_label)
        self.shown_for_button[helix_button].add(structure_label)
        self.all_params_widgets.append(structure_label)
        from chimerax.atomic.widgets import StructureListWidget
        class ShorterStructureListWidget(StructureListWidget):
            def sizeHint(self):
                size = super().sizeHint()
                size.setHeight(size.height()//2)
                return size
        self.helix_structure_list = ShorterStructureListWidget(self.session)
        params_layout.addWidget(self.helix_structure_list)
        params_layout.setStretch(params_layout.count()-1, 1)
        self.shown_for_button[helix_button].add(self.helix_structure_list)
        self.all_params_widgets.append(self.helix_structure_list)

        params_group = QGroupBox("Axis Parameters")
        params_layout.addWidget(params_group)
        pg_layout = QGridLayout()
        pg_layout.setColumnMinimumWidth(1, 9)
        pg_layout.setColumnStretch(2, 1)
        pg_layout.setSpacing(0)
        pg_layout.setContentsMargins(1,1,1,1)
        params_group.setLayout(pg_layout)

        from itertools import count
        row_count = count(0)
        row = next(row_count)
        pg_layout.setRowStretch(row, 1)

        row = next(row_count)
        color_label = QLabel("Color")
        pg_layout.addWidget(color_label, row, 0, alignment=Qt.AlignRight)
        color_layout = QHBoxLayout()
        color_layout.setSpacing(0)
        pg_layout.addLayout(color_layout, row, 2, alignment=Qt.AlignLeft)

        row = next(row_count)
        pg_layout.setRowStretch(row, 1)

        self.color_group = QButtonGroup()
        self.default_color_button = QRadioButton("default")
        self.color_group.addButton(self.default_color_button)
        color_layout.addWidget(self.default_color_button)
        color_layout.addSpacing(9)
        explicit_color_button = QRadioButton()
        self.color_group.addButton(explicit_color_button)
        color_layout.addWidget(explicit_color_button, alignment=Qt.AlignRight)
        self.color_widget = ColorButton(max_size=(16,16))
        self.color_widget.color_changed.connect(
            lambda *args, but=explicit_color_button: but.setChecked(True))
        from chimerax.core.colors import Color
        # button doesn't start off the right size unless explicitly given a color
        self.color_widget.color = Color("#909090")
        color_layout.addWidget(self.color_widget, alignment=Qt.AlignLeft)
        self.default_color_button.setChecked(True)

        row = next(row_count)
        name_label = QLabel("Name")
        pg_layout.addWidget(name_label, row, 0, alignment=Qt.AlignRight)
        self.name_entry = QLineEdit()
        pg_layout.addWidget(self.name_entry, row, 2)

        row = next(row_count)
        pg_layout.setRowStretch(row, 1)

        row = next(row_count)
        radius_label = QLabel("Radius")
        pg_layout.addWidget(radius_label, row, 0, alignment=Qt.AlignRight)
        radius_layout = QHBoxLayout()
        radius_layout.setSpacing(0)
        pg_layout.addLayout(radius_layout, row, 2, alignment=Qt.AlignLeft)

        row = next(row_count)
        pg_layout.setRowStretch(row, 1)

        self.radius_group = QButtonGroup()
        self.default_radius_button = QRadioButton("default")
        self.radius_group.addButton(self.default_radius_button)
        radius_layout.addWidget(self.default_radius_button)
        radius_layout.addSpacing(9)
        explicit_radius_button = QRadioButton()
        self.radius_group.addButton(explicit_radius_button)
        radius_layout.addWidget(explicit_radius_button, alignment=Qt.AlignRight)
        explicit_radius_layout = QHBoxLayout()
        radius_layout.addLayout(explicit_radius_layout, stretch=1)
        self.radius_entry = QLineEdit()
        self.radius_entry.setValidator(QDoubleValidator())
        self.radius_entry.textChanged.connect(
            lambda *args, but=explicit_radius_button: but.setChecked(True))
        radius_layout.addWidget(self.radius_entry, alignment=Qt.AlignLeft, stretch=1)
        radius_layout.addWidget(QLabel(" \N{ANGSTROM SIGN}"))
        self.default_radius_button.setChecked(True)

        self.show_applicable_params(self.button_group.checkedButton())

        bbox = qbbox(qbbox.Ok | qbbox.Apply | qbbox.Close | qbbox.Help)
        bbox.accepted.connect(self.define_axis)
        bbox.rejected.connect(lambda tw=tw: setattr(tw, 'shown', False))
        bbox.button(qbbox.Apply).clicked.connect(lambda *args: self.define_axis(hide=False))
        bbox.button(qbbox.Help).setEnabled(False)
        layout.addWidget(bbox)

        tw.manage(None)

    cmd_params_plane_axis = cmd_params_points_axis = None

    def cmd_params_atoms_axis(self):
        from chimerax.atomic import selected_atoms
        if not selected_atoms(self.session):
            raise UserError("No atoms/centroids selected")
        return "sel " + self.generic_params()

    def cmd_params_helix_axis(self):
        structures = self.helix_structure_list.value
        if not structures:
            raise UserError("No structures chosen")
        from chimerax.core.commands import concise_model_spec
        from chimerax.atomic import Structure
        params = concise_model_spec(self.session, structures, relevant_types=Structure)
        if params:
            params += " "

        return params + "perHelix true " + self.generic_params()

    def generic_params(self):
        from chimerax.core.commands import StringArg
        color_button = self.color_group.checkedButton()
        if color_button == self.default_color_button:
            params = ""
        else:
            from chimerax.core.colors import color_name
            params = "color " + StringArg.unparse(color_name(self.color_widget.color)) + " "

        name = self.name_entry.text().strip()
        if name:
            params += "name %s " % StringArg.unparse(name)

        radius_button = self.radius_group.checkedButton()
        if radius_button != self.default_radius_button:
            if not self.radius_entry.hasAcceptableInput():
                raise UserError("Radius must be a number")
            radius = float(self.radius_entry.text())
            if radius <= 0.0:
                raise UserError("Radius must be a positive number")
            params += "radius %g" % radius
        return params

    def define_axis(self, hide=True):
        # gather parameters here, so that dialog stays up if there's an error in parameters
        cmd_params = self.button_dispatch[self.button_group.checkedButton()]()
        if hide:
            self.tool_window.shown = False
        # run command here
        from chimerax.core.commands import run
        run(self.session, "define axis " + cmd_params)

    def show_applicable_params(self, button):
        # widgets that are _always_ shown aren't in "shown_widgets"
        shown_widgets = self.shown_for_button[button]
        for widget in self.all_params_widgets:
            widget.setHidden(widget not in shown_widgets)
        self.name_entry.setText(self.axis_name_for_button[button])
