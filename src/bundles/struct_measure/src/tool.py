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

from chimerax.core.errors import UserError, LimitationError
from chimerax.core.tools import ToolInstance
from chimerax.core.commands import concise_model_spec, run, StringArg
from Qt.QtWidgets import QTableWidget, QHBoxLayout, QVBoxLayout, QAbstractItemView, QWidget, QPushButton, \
    QTabWidget, QTableWidgetItem, QFileDialog, QDialogButtonBox as qbbox, QLabel, QButtonGroup, \
    QRadioButton, QLineEdit, QGroupBox, QGridLayout, QCheckBox, QMenu
from Qt.QtGui import QDoubleValidator
from Qt.QtCore import Qt
from chimerax.ui.widgets import ColorButton
from chimerax.ui.options import SettingsPanel, OptionsPanel, Option, \
    BooleanOption, ColorOption, IntOption, FloatOption, StringOption
from chimerax.centroids import CentroidModel
from chimerax.axes_planes import AxisModel, PlaneModel

class StructMeasureTool(ToolInstance):

    tab_names = ["Distances", "Angles/Torsions", "Axes/Planes/Centroids"]
    help_info = ["distances", "angles", None]

    def __init__(self, session):
        ToolInstance.__init__(self, session, "Structure Measurements")
        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area
        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self._tab_changed_cb)
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

    def _apc_align_and_center(self):
        sel = self.apc_table.selected
        from chimerax.axes_planes import AxisModel, PlaneModel
        if len(sel) != 1 or not isinstance(sel[0], (AxisModel, PlaneModel)):
            raise UserError("Choose exactly one axis or plane in table")
        item = sel[0]
        run(self.session, "view zalign %s" % item.atomspec)
        axis = self.apc_axis_button.text()
        if axis == 'Y':
            run(self.session, "turn x 90 center %s" % item.atomspec)
        elif axis == 'X':
            run(self.session, "turn y 90 center %s" % item.atomspec)

    def _apc_delete_items(self):
        items = self.apc_table.selected
        if not items:
            items = self.apc_table.data
            if not items:
                raise UserError("No items in table")
            if len(items) > 1:
                from chimerax.ui.ask import ask
                if ask(self.session, "Really delete %d items?" % len(items)) == "no":
                    return
        run(self.session, "close %s" % concise_model_spec(self.session, items))

    def _apc_report_distance(self):
        sel = self.apc_table.selected
        if not sel:
            raise UserError("No items chosen in table")
        from chimerax.atomic import selected_atoms
        sel_atoms = selected_atoms(self.session)
        if not sel_atoms:
            raise UserError("No atoms selected")
        # Try to "promote" atoms to higher "level" to make shorter atom spec
        residues = sel_atoms.unique_residues
        if sum(residues.num_atoms) == len(sel_atoms):
            chains = residues.unique_chains
            if sum(chains.num_existing_residues) == len(residues):
                structures = chains.structures.unique()
                if sum(structures.num_chains) == len(chains):
                    spec_source = structures
                else:
                    spec_source = chains
            else:
                spec_source = residues
        else:
            spec_source = sel_atoms
        target_spec = "".join([x.atomspec for x in spec_source])
        item_spec = "".join([x.atomspec for x in sel])
        info = run(self.session, "distance %s %s" % (target_spec, item_spec))
        if len(sel) == 1:
            min_info, avg, max_info = list(info.values())[0]
            dist_fmt = self.session.pb_dist_monitor.distance_format
            self.apc_status_label.setText(("Distance from %s to %d atoms: min " + dist_fmt
                + " (%s), avg " + dist_fmt + ", max " + dist_fmt + " (%s)")
                % (sel[0], len(sel_atoms), min_info[0], min_info[1], avg, max_info[0], max_info[1]))
            self.apc_status_label.setHidden(False)

    def _apc_save_info(self):
        apc_items = self.apc_table.data
        if not apc_items:
            raise UserError("No axes/planes/centroids to save!")
        path = QFileDialog.getSaveFileName(self.angle_table, "Save Axes/Planes/Centroids Info File")[0]
        if not path:
            return
        need_space = False
        with open(path, 'wt') as f:
            for section_hdr, model_type, formatting in [
                ("Axes\nname, model ID, length, center, direction", AxisModel,
                    lambda x: "%s: %s %g %s %s" % (x.name, x.atomspec, x.length, x.center, x.direction)),
                ("Planes\nname, model ID, center, normal, radius", PlaneModel,
                    lambda x: "%s: %s %s %s %g" % (x.name, x.atomspec, x.center, x.normal, x.radius)),
                ("Centroids\nname, model ID, center", CentroidModel,
                    lambda x: "%s: %s %s" % (x.name, x.atomspec, x.atoms[0].coord)),
            ]:
                items = [x for x in apc_items if isinstance(x, model_type)]
                if not items:
                    continue
                if need_space:
                    print("", file=f)
                print(section_hdr, file=f)
                for item in items:
                    print(formatting(item), file=f)
                need_space = True

    def _apc_selection_changed(self, newly_selected, newly_deselected):
        sel = self.apc_table.selected
        if len(sel) == 2:
            sel1, sel2 = sel
            from chimerax.dist_monitor import ComplexMeasurable
            comp1, comp2 = [isinstance(x, ComplexMeasurable) for x in sel]
            if comp1 or comp2:
                dist = angle = None
                if comp1:
                    d = sel1.distance(sel2)
                    if d is not NotImplemented:
                        dist = d
                    a = sel1.angle(sel2)
                    if a is not NotImplemented:
                        angle = a
                if comp2:
                    if dist is None:
                        d = sel2.distance(sel1)
                        if d is not NotImplemented:
                            dist = d
                    if angle is None:
                        a = sel2.angle(sel1)
                        if a is not notImplemented:
                            angle = a
            else:
                from chimerax.geometry import distance
                dist = distance(sel1.scene_coord, sel2.scene_coord)
                angle = None
            if dist is None and angle is None:
                info = ["no distance/angle"]
            else:
                info = []
                if dist is not None:
                    dist_fmt = self.session.pb_dist_monitor.distance_format
                    info.append("distance: " + dist_fmt % dist)
                if angle is not None:
                    info.append("angle: %.3f" % angle)
            info_text = "; ".join(info)
            self.apc_status_label.setText(info_text)
            self.apc_status_label.setHidden(False)
            self.session.logger.info("<b>%s</b> <i>to</i> <b>%s</b>: %s" % (sel1, sel2, info_text),
                is_html=True)
        elif newly_selected or newly_deselected:
            self.apc_status_label.setHidden(True)

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
        run(self.session, "distance %s %s" % tuple(a.string(style="command") for a in sel_atoms))

    def _delete_angle(self):
        rows = set([index.row() for index in self.angle_table.selectedIndexes()])
        if not rows:
            rows = range(self.angle_table.rowCount())
            if len(rows) > 1:
                from chimerax.ui.ask import ask
                if ask(self.session, "Really delete %d angles/torsions?" % len(rows)) == "no":
                    return
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
            rows = range(self.dist_table.rowCount())
            if len(rows) > 1:
                from chimerax.ui.ask import ask
                if ask(self.session, "Really delete %d distances?" % len(rows)) == "no":
                    return
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
        self.apc_status_tip = "Choose two items in table to report angle/distance (also logged).\n"\
            "Double click Name/ID to edit."
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        tab_area.setLayout(layout)

        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)
        axis_button = QPushButton("Define axes...")
        axis_button.clicked.connect(lambda *args, cb=self._show_define_axis_dialog: cb())
        button_layout.addWidget(axis_button, alignment=Qt.AlignHCenter)
        plane_button = QPushButton("Define plane...")
        plane_button.clicked.connect(lambda *args, cb=self._show_define_plane_dialog: cb())
        button_layout.addWidget(plane_button, alignment=Qt.AlignHCenter)
        centroid_button = QPushButton("Define centroid...")
        centroid_button.clicked.connect(lambda *args, cb=self._show_define_centroid_dialog: cb())
        button_layout.addWidget(centroid_button, alignment=Qt.AlignHCenter)
        self._define_axis_dialog = self._define_plane_dialog = self._define_centroid_dialog = None

        table_area = QWidget()
        layout.addWidget(table_area, stretch=1)
        table_layout = QVBoxLayout()
        table_layout.setSpacing(1)
        table_area.setLayout(table_layout)
        from chimerax.ui.widgets import ItemTable
        self.apc_table = ItemTable(session=self.session)
        def verify_id(id_text, *, ses=self.session):
            try:
                ids = [int(x) for x in id_text.split('.')]
            except Exception:
                ses.logger.error("ID must be one or more integers separated by '.' characters")
                return False
            return True
        self.apc_table.add_column("Name", "name", data_set="rename {item.atomspec} {value}",
            validator=lambda name: not name.isspace())
        def id_cmp(item1, item2):
            components1 = [int(x) for x in item1.id_string.split('.')]
            components2 = [int(x) for x in item2.id_string.split('.')]
            for i in range(max(len(components1), len(components2))):
                if components1[i] == components2[i]:
                    continue
                return components1[i] < components2[i]
            return len(components1) < len(components2)
        self.apc_table.add_column("ID", "id_string", data_set="rename {item.atomspec} id #{value}",
            validator=verify_id, sort_func=id_cmp)
        self.apc_table.add_column("Color", "model_color", format=ItemTable.COL_FORMAT_TRANSPARENT_COLOR,
            data_set="color {item.atomspec} {value}", title_display=False)
        self.apc_table.add_column("Shown", "display", format=ItemTable.COL_FORMAT_BOOLEAN, icon="shown")
        def run_sel_cmd(item, value, ses=self.session):
            cmd = "select" if value else "~select"
            run(ses, cmd + " " + item.atomspec)
        self.apc_table.add_column("Selected", "selected", format=ItemTable.COL_FORMAT_BOOLEAN, icon="select",
            data_set=run_sel_cmd, title_display=False)
        self.apc_table.add_column("Length", "length", format="%4.1f")
        self.apc_table.add_column("Radius", "radius", format="%4.1f")
        self.apc_table.launch()
        self.apc_table.data = self._filter_apc_models(self.session.models)
        self.apc_table.sortByColumn(1, Qt.AscendingOrder)
        self.apc_table.selection_changed.connect(self._apc_selection_changed)
        from chimerax.core.models import ADD_MODELS, REMOVE_MODELS
        self.handlers.append(self.session.triggers.add_handler(ADD_MODELS, self._refresh_apc_table))
        self.handlers.append(self.session.triggers.add_handler(REMOVE_MODELS, self._refresh_apc_table))
        from chimerax.core.models import MODEL_COLOR_CHANGED, MODEL_DISPLAY_CHANGED, MODEL_ID_CHANGED, \
            MODEL_NAME_CHANGED, MODEL_SELECTION_CHANGED
        for trig_name in  (MODEL_COLOR_CHANGED, MODEL_DISPLAY_CHANGED, MODEL_ID_CHANGED, MODEL_NAME_CHANGED,
                MODEL_SELECTION_CHANGED):
            self.handlers.append(self.session.triggers.add_handler(trig_name, self._refresh_apc_cell))
        table_layout.addWidget(self.apc_table, alignment=Qt.AlignHCenter)
        self.apc_status_label = QLabel(self.apc_status_tip)
        self.apc_status_label.setWordWrap(True)
        self.apc_status_label.setAlignment(Qt.AlignHCenter)
        table_layout.addWidget(self.apc_status_label)
        delete_layout = QHBoxLayout()
        delete_layout.setSpacing(0)
        delete_layout.setContentsMargins(0,0,0,0)
        table_layout.addLayout(delete_layout)
        delete_button = QPushButton("Delete")
        delete_button.clicked.connect(self._apc_delete_items)
        delete_button.setToolTip("Delete items selected in table (or all if none selected)")
        delete_layout.addWidget(delete_button, alignment=Qt.AlignCenter)
        info_button = QPushButton("Save Info...")
        info_button.clicked.connect(self._apc_save_info)
        delete_layout.addWidget(info_button, alignment=Qt.AlignCenter)
        report_layout_widget = QWidget() # so that the label+button is centered collectively
        report_layout = QHBoxLayout()
        report_layout.setSpacing(0)
        report_layout.setContentsMargins(0,0,0,0)
        report_layout_widget.setLayout(report_layout)
        table_layout.addWidget(report_layout_widget, alignment=Qt.AlignCenter)
        report_button = QPushButton("Report distance")
        report_button.clicked.connect(self._apc_report_distance)
        report_layout.addWidget(report_button, alignment=Qt.AlignRight)
        report_layout.addWidget(QLabel(" to selected atoms"), alignment=Qt.AlignLeft)
        align_layout_widget = QWidget() # so that the widgets are centered collectively
        align_layout = QHBoxLayout()
        align_layout.setSpacing(0)
        align_layout.setContentsMargins(0,0,0,0)
        align_layout_widget.setLayout(align_layout)
        table_layout.addWidget(align_layout_widget, alignment=Qt.AlignCenter)
        align_button = QPushButton("Center and align")
        align_button.clicked.connect(self._apc_align_and_center)
        align_layout.addWidget(align_button)
        align_layout.addWidget(QLabel(" chosen axis or plane normal along "))
        self.apc_axis_button = QPushButton("X")
        menu = QMenu(self.apc_axis_button)
        for axis in ['X', 'Y', 'Z']:
            menu.addAction(axis)
        menu.triggered.connect(lambda action, but=self.apc_axis_button: but.setText(action.text()))
        self.apc_axis_button.setMenu(menu)
        align_layout.addWidget(self.apc_axis_button)

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

        panel = SettingsPanel()
        from chimerax.dist_monitor.settings import settings
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

    def _filter_apc_models(self, models):
        return [m for m in models if isinstance(m, (CentroidModel, AxisModel, PlaneModel))]

    def _refresh_apc_cell(self, trig_name, model):
        if not self._filter_apc_models([model]):
            return
        from chimerax.core.models import MODEL_COLOR_CHANGED, MODEL_DISPLAY_CHANGED, MODEL_ID_CHANGED, \
            MODEL_NAME_CHANGED
        if trig_name == MODEL_COLOR_CHANGED:
            title = "Color"
        elif trig_name == MODEL_DISPLAY_CHANGED:
            title = "Shown"
        elif trig_name == MODEL_ID_CHANGED:
            title = "ID"
        elif trig_name == MODEL_NAME_CHANGED:
            title = "Name"
        else:
            title = "Selected"
        self.apc_table.update_cell(title, model)


    def _refresh_apc_table(self, trig_name, models):
        if self._filter_apc_models(models):
            self.apc_table.data = self._filter_apc_models(self.session.models)
            columns = self.apc_table.columns
            for i, col in enumerate(columns):
                if col.display_format not in self.apc_table.color_formats:
                    self.apc_table.resizeColumnToContents(i)

    def _save_angle_info(self):
        if not self._angle_info:
            raise UserError("No angles/torsions to save!")
        path = QFileDialog.getSaveFileName(self.angle_table, "Save Angles/Torsions File")[0]
        if not path:
            return
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
        if not self._define_centroid_dialog:
            self._define_centroid_dialog = DefineCentroidDialog(self)
        self._define_centroid_dialog.tool_window.shown = True

    def _show_define_plane_dialog(self):
        if not self._define_plane_dialog:
            self._define_plane_dialog = DefinePlaneDialog(self)
        self._define_plane_dialog.tool_window.shown = True

    def _tab_changed_cb(self, index):
        self._set_help(index)
        if self.tab_names[index] == "Axes/Planes/Centroids":
            if self.apc_status_label.text() == self.apc_status_tip:
                def clear_status(s=self):
                    if s.apc_status_label.text() == s.apc_status_tip:
                        s.apc_status_label.setHidden(True)
                from Qt.QtCore import QTimer
                self._hold_ref_to_timer = QTimer.singleShot(12000, clear_status)

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

class AngstromOption(FloatOption):
    def __init__(self, *args, **kw):
        kw['right_text'] = "\N{ANGSTROM SIGN}"
        super().__init__(*args, **kw)

class DefineAxisDialog:
    def __init__(self, sm_tool):
        self.tool_window = tw = sm_tool.tool_window.create_child_window("Define Axes", close_destroys=False)
        self.session = sm_tool.session
        layout = QVBoxLayout()
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

        helix_button = QRadioButton("Each helix in structure")
        helix_button.setChecked(True)
        self.button_group.addButton(helix_button)
        type_layout.addWidget(helix_button, alignment=Qt.AlignLeft | Qt.AlignTop)
        self.shown_for_button[helix_button] = set()
        self.axis_name_for_button[helix_button] = "helix axes"
        self.button_dispatch[helix_button] = self.cmd_params_helix_axis

        atoms_button = QRadioButton("Selected atoms/centroids")
        atoms_button.setChecked(False)
        self.button_group.addButton(atoms_button)
        type_layout.addWidget(atoms_button, alignment=Qt.AlignLeft | Qt.AlignTop)
        self.shown_for_button[atoms_button] = set()
        self.axis_name_for_button[atoms_button] = "axis"
        self.button_dispatch[atoms_button] = self.cmd_params_atoms_axis

        plane_button = QRadioButton("Plane normals")
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

        structure_label = QLabel("Structures")
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

        plane_label = QLabel("Planes")
        params_layout.addWidget(plane_label)
        self.shown_for_button[plane_button].add(plane_label)
        self.all_params_widgets.append(plane_label)
        from chimerax.ui.widgets import ModelListWidget
        class ShorterModelListWidget(ModelListWidget):
            def sizeHint(self):
                size = super().sizeHint()
                size.setHeight(size.height()//2)
                return size
        from chimerax.axes_planes import PlaneModel
        self.plane_list = ShorterModelListWidget(self.session, class_filter=PlaneModel)
        params_layout.addWidget(self.plane_list)
        params_layout.setStretch(params_layout.count()-1, 1)
        self.shown_for_button[plane_button].add(self.plane_list)
        self.all_params_widgets.append(self.plane_list)

        xyz_widget = QWidget()
        params_layout.addWidget(xyz_widget, alignment=Qt.AlignCenter)
        self.shown_for_button[points_button].add(xyz_widget)
        self.all_params_widgets.append(xyz_widget)
        points_layout = QGridLayout()
        points_layout.setSpacing(1)
        xyz_widget.setLayout(points_layout)
        for i, lab in enumerate(["X", "Y", "Z"]):
            points_layout.addWidget(QLabel(lab), 0, i+1, alignment=Qt.AlignBottom | Qt.AlignHCenter)
        self.xyz_widgets = []
        for xyz_row in range(2):
            points_layout.addWidget(QLabel(["from", "to"][xyz_row]), xyz_row+1, 0, alignment=Qt.AlignRight)
            row_widgets = []
            for col in range(3):
                widget = QLineEdit()
                widget.setValidator(QDoubleValidator())
                widget.setAlignment(Qt.AlignCenter)
                widget.setMaximumWidth(50)
                points_layout.addWidget(widget, xyz_row+1, col+1)
                row_widgets.append(widget)
            self.xyz_widgets.append(row_widgets)

        params_group = QGroupBox("Axis Parameters")
        params_layout.addWidget(params_group)
        pg_layout = QHBoxLayout()
        params_group.setLayout(pg_layout)
        self.options = OptionsPanel(sorting=False, scrolled=False)
        pg_layout.addWidget(self.options)
        self.color_option = ColorWithDefaultOption("Color", None, None, has_alpha_channel=True)
        self.options.add_option(self.color_option)
        self.name_option = StringOption("Name", None, None)
        self.options.add_option(self.name_option)
        self.radius_type_option = BooleanOption("Set radius to average atom-axis distance", True,
            lambda opt, s=self: s.options.set_option_shown(s.radius_option, not opt.value))
        self.options.add_option(self.radius_type_option)
        self.radius_option = AngstromOption("Fixed radius", 2.0, None, min="positive", decimal_places=1,
            step=1.0)
        self.options.add_option(self.radius_option)
        self.options.hide_option(self.radius_option)
        self.length_type_option = BooleanOption("Set length to enclose atom projections", True,
            lambda opt, s=self: (s.options.set_option_shown(s.length_option, not opt.value),
            s.options.set_option_shown(s.padding_option, opt.value)))
        self.options.add_option(self.length_type_option)
        self.length_option = AngstromOption("Fixed length", 10.0, None, min="positive", decimal_places=1,
            step=1.0)
        self.options.add_option(self.length_option)
        self.options.hide_option(self.length_option)
        self.padding_option = AngstromOption("with padding", 0.0, None, decimal_places=1, step=0.5)
        self.options.add_option(self.padding_option)
        self.weighting_option = BooleanOption("Mass weighting", False, None)
        self.options.add_option(self.weighting_option)
        self.shown_for_button[atoms_button].add(self.weighting_option)
        self.all_params_widgets.append(self.weighting_option)

        """
        self.params_group_layout = pg_layout = QGridLayout()
        pg_layout.setColumnMinimumWidth(1, 9)
        pg_layout.setColumnStretch(2, 1)
        pg_layout.setSpacing(0)
        pg_layout.setContentsMargins(1,1,1,1)
        params_group.setLayout(pg_layout)

        from itertools import count
        row_count = count(0)
        # blank "spacer" rows above, below, and between
        row = next(row_count)
        pg_layout.setRowStretch(row, 1)

        row = next(row_count)
        pg_layout.addWidget(QLabel("Color"), row, 0, alignment=Qt.AlignRight)
        color_layout = QHBoxLayout()
        color_layout.setSpacing(0)
        pg_layout.addLayout(color_layout, row, 2, alignment=Qt.AlignLeft)

        row = next(row_count)
        pg_layout.setRowStretch(row, 1)

        self.color_group = QButtonGroup()
        self.default_color_button = QRadioButton("atom-based")
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
        pg_layout.addWidget(QLabel("Name"), row, 0, alignment=Qt.AlignRight)
        self.name_entry = QLineEdit()
        pg_layout.addWidget(self.name_entry, row, 2)

        row = next(row_count)
        pg_layout.setRowStretch(row, 1)

        row = next(row_count)
        pg_layout.addWidget(QLabel("Radius"), row, 0, alignment=Qt.AlignRight)
        radius_layout = QHBoxLayout()
        radius_layout.setSpacing(0)
        pg_layout.addLayout(radius_layout, row, 2, alignment=Qt.AlignLeft)

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
        self.radius_entry.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.radius_entry.textChanged.connect(
            lambda *args, but=explicit_radius_button: but.setChecked(True))
        radius_layout.addWidget(self.radius_entry, alignment=Qt.AlignLeft, stretch=1)
        radius_layout.addWidget(QLabel(" \N{ANGSTROM SIGN}"))
        self.default_radius_button.setChecked(True)

        row = next(row_count)
        pg_layout.setRowStretch(row, 1)

        row = next(row_count)
        pg_layout.addWidget(QLabel("Length"), row, 0, alignment=Qt.AlignRight)
        length_layout = QHBoxLayout()
        length_layout.setSpacing(0)
        pg_layout.addLayout(length_layout, row, 2, alignment=Qt.AlignLeft)

        self.length_group = QButtonGroup()
        self.default_length_button = QRadioButton("default")
        self.length_group.addButton(self.default_length_button)
        length_layout.addWidget(self.default_length_button)
        length_layout.addSpacing(9)
        explicit_length_button = QRadioButton()
        self.length_group.addButton(explicit_length_button)
        length_layout.addWidget(explicit_length_button, alignment=Qt.AlignRight)
        explicit_length_layout = QHBoxLayout()
        length_layout.addLayout(explicit_length_layout, stretch=1)
        self.length_entry = QLineEdit()
        self.length_entry.setValidator(QDoubleValidator())
        self.length_entry.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        length_layout.addWidget(self.length_entry, alignment=Qt.AlignLeft, stretch=1)
        length_layout.addWidget(QLabel(" \N{ANGSTROM SIGN}"))
        row = next(row_count)
        padding_layout = QHBoxLayout()
        pg_layout.addLayout(padding_layout, row, 2, alignment=Qt.AlignLeft)
        padding_widgets = []
        padding_label = QLabel("padding:")
        padding_layout.addWidget(padding_label)
        padding_widgets.append(padding_label)
        self.padding_entry = QLineEdit()
        self.padding_entry.setValidator(QDoubleValidator())
        self.padding_entry.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.padding_entry.setText("0")
        self.padding_entry.setMaximumWidth(30)
        padding_layout.addWidget(self.padding_entry, alignment=Qt.AlignLeft)
        padding_widgets.append(self.padding_entry)
        padding_angstroms = QLabel(" \N{ANGSTROM SIGN}")
        padding_layout.addWidget(padding_angstroms, alignment=Qt.AlignLeft)
        padding_widgets.append(padding_angstroms)
        self.length_entry.textChanged.connect(lambda *args, but=explicit_length_button,
            widgets=padding_widgets: (but.setChecked(True), [w.setHidden(True) for w in widgets]))
        explicit_length_button.clicked.connect(lambda *args, widgets=padding_widgets:
            [w.setHidden(True) for w in widgets])
        self.default_length_button.clicked.connect(lambda *args, widgets=padding_widgets:
            [w.setHidden(False) for w in widgets])
        self.default_length_button.setChecked(True)

        row = next(row_count)
        pg_layout.setRowStretch(row, 1)

        row = next(row_count)
        self.mass_weighting = QCheckBox("Mass weighting")
        self.mass_weighting.setChecked(False)
        pg_layout.addWidget(self.mass_weighting, row, 0, 1, 3, alignment=Qt.AlignCenter)
        self.shown_for_button[atoms_button].add((self.mass_weighting, row+1))
        self.all_params_widgets.append((self.mass_weighting, row+1))

        row = next(row_count)
        pg_layout.setRowStretch(row, 1)
        """

        self.show_applicable_params(self.button_group.checkedButton())

        bbox = qbbox(qbbox.Ok | qbbox.Apply | qbbox.Close | qbbox.Help)
        bbox.accepted.connect(self.define_axis)
        bbox.rejected.connect(lambda tw=tw: setattr(tw, 'shown', False))
        bbox.button(qbbox.Apply).clicked.connect(lambda *args: self.define_axis(hide=False))
        bbox.button(qbbox.Help).setEnabled(False)
        layout.addWidget(bbox)

        tw.manage(None)

    def cmd_params_atoms_axis(self):
        from chimerax.atomic import selected_atoms
        if not selected_atoms(self.session):
            raise UserError("No atoms/centroids selected")
        base_params = "sel " + self.generic_params()
        #TODO
        if self.mass_weighting.isChecked():
            return base_params + " mass true"
        return base_params

    def cmd_params_helix_axis(self):
        structures = self.helix_structure_list.value
        if not structures:
            raise UserError("No structures chosen")
        from chimerax.atomic import Structure
        params = concise_model_spec(self.session, structures, relevant_types=Structure)
        if params:
            params += " "

        return params + "perHelix true " + self.generic_params()

    def cmd_params_plane_axis(self):
        planes = self.plane_list.value
        if not planes:
            raise UserError("No planes chosen")
        params = concise_model_spec(self.session, planes)
        if params:
            params += " "

        return params + self.generic_params()

    def cmd_params_points_axis(self):
        processed_data = []
        for row_widgets in self.xyz_widgets:
            processed_row = []
            for entry in row_widgets:
                if not entry.text():
                    raise UserError("All XYZ fields must have numbers!")
                if not entry.hasAcceptableInput():
                    raise UserError("All XYZ fields must contain numbers")
                processed_row.append(float(entry.text()))
            processed_data.append(processed_row)
        from_pt, to_pt = processed_data
        return ("fromPoint %g,%g,%g toPoint %g,%g,%g " % (*from_pt, *to_pt))+ self.generic_params()

    def generic_params(self):
        #TODO
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
            params += "radius %g " % radius

        length_button = self.length_group.checkedButton()
        if length_button == self.default_length_button:
            text = self.padding_entry.text()
            if text:
                if not self.padding_entry.hasAcceptableInput():
                    raise UserError("Padding must be a number")
                padding = float(self.padding_entry.text())
                if padding != 0.0:
                    params += "padding %g " % padding
        else:
            if not self.length_entry.hasAcceptableInput():
                raise UserError("Length must be a number")
            length = float(self.length_entry.text())
            if length <= 0.0:
                raise UserError("Length must be a positive number")
            params += "length %g " % length
        return params

    def define_axis(self, hide=True):
        # gather parameters here, so that dialog stays up if there's an error in parameters
        cmd_params = self.button_dispatch[self.button_group.checkedButton()]()
        if hide:
            self.tool_window.shown = False
        # run command here
        run(self.session, "define axis " + cmd_params)

    def show_applicable_params(self, button):
        # widgets that are _always_ shown aren't in "shown_widgets"
        shown_widgets = self.shown_for_button[button]
        for widget in self.all_params_widgets:
            hidden = widget not in shown_widgets
            if isinstance(widget, Option):
                self.options.set_option_shown(widget, not hidden)
            else:
                widget.setHidden(hidden)
        self.name_option.value = self.axis_name_for_button[button]

class ColorWithDefaultOption(Option):
    def set_multiple(self):
        self.value = None

    def get_value(self):
        if self.color_group.checkedButton() == self.default_color_button:
            return None
        return self.color_widget.color

    def set_value(self, value):
        if value is None:
            self.default_color_button.setChecked(True)
        else:
            self.default_color_button.setChecked(False)
            self.color_widget.color = value

    value = property(get_value, set_value)

    def _make_widget(self, **kw):
        self.widget = QWidget()
        layout = QHBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0,0,0,0)
        self.widget.setLayout(layout)
        self.color_group = QButtonGroup()
        self.default_color_button = QRadioButton("atom-based")
        self.color_group.addButton(self.default_color_button)
        layout.addWidget(self.default_color_button)
        layout.addSpacing(9)
        explicit_color_button = QRadioButton()
        self.color_group.addButton(explicit_color_button)
        layout.addWidget(explicit_color_button, alignment=Qt.AlignRight)
        self.color_widget = ColorButton(max_size=(16,16), **kw)
        self.color_widget.color_changed.connect(
            lambda *args, but=explicit_color_button: but.setChecked(True))
        from chimerax.core.colors import Color
        # button doesn't start off the right size unless explicitly given a color
        self.color_widget.color = Color("#909090")
        layout.addWidget(self.color_widget, alignment=Qt.AlignLeft)
        self.default_color_button.setChecked(True)

class DefinePlaneDialog:
    def __init__(self, sm_tool):
        self.tool_window = tw = sm_tool.tool_window.create_child_window("Define Plane", close_destroys=False)
        self.session = sm_tool.session
        layout = QVBoxLayout()
        layout.setSpacing(0)
        tw.ui_area.setLayout(layout)

        layout.addWidget(QLabel("Create plane for selected atoms..."), alignment=Qt.AlignCenter)
        self.options_panel = options_panel = OptionsPanel(sorting=False, scrolled=False)
        layout.addWidget(options_panel)
        self.name_option = StringOption("Plane name", "plane", None)
        options_panel.add_option(self.name_option)
        self.color_option = ColorWithDefaultOption("Color", None, None, has_alpha_channel=True)
        options_panel.add_option(self.color_option)
        self.enclose_option = BooleanOption("Set disk radius to enclose atom projections", True,
            self._enclosed_changed)
        options_panel.add_option(self.enclose_option)
        self.padding_option = AngstromOption("Extra radius (padding)", 0.0, None, decimal_places=1, step=1.0)
        options_panel.add_option(self.padding_option)
        self.radius_option = AngstromOption("Fixed radius", 10.0, None, decimal_places=1, step=1.0,
            min="positive")
        options_panel.add_option(self.radius_option)
        options_panel.hide_option(self.radius_option)
        self.thickness_option = AngstromOption("Disk thickness", 0.1, None, decimal_places=2, step=.05,
            min="positive")
        options_panel.add_option(self.thickness_option)

        bbox = qbbox(qbbox.Ok | qbbox.Apply | qbbox.Close | qbbox.Help)
        bbox.accepted.connect(self.define_plane)
        bbox.rejected.connect(lambda tw=tw: setattr(tw, 'shown', False))
        bbox.button(qbbox.Apply).clicked.connect(lambda *args: self.define_plane(hide=False))
        bbox.button(qbbox.Help).setEnabled(False)
        layout.addWidget(bbox)

        tw.manage(None)

    def define_plane(self, hide=True):
        # gather parameters before hiding, so that dialog stays up if there's an error in parameters
        from chimerax.atomic import selected_atoms
        sel_atoms = selected_atoms(self.session)
        if len(sel_atoms) < 3:
            raise UserError("Need to select at least 3 atoms to define a plane")
        cmd = "define plane sel"
        plane_name = self.name_option.value
        if plane_name and plane_name != "plane":
            cmd += " name %s" % StringArg.unparse(plane_name)
        color = self.color_option.value
        if color is not None:
            from chimerax.core.colors import color_name
            cmd += " color " + StringArg.unparse(color_name(color))
        enclose = self.enclose_option.value
        if enclose:
            padding = self.padding_option.value
            if padding != 0.0:
                cmd += " padding %g" % padding
        else:
            radius = self.radius_option.value
            cmd += " radius %g" % radius
        thickness = self.thickness_option.value
        if thickness != 0.1:
            cmd += " thickness %g" % thickness
        if hide:
            self.tool_window.shown = False
        # run command here
        run(self.session, cmd)

    def _enclosed_changed(self, opt):
        if opt.value:
            self.options_panel.hide_option(self.radius_option)
            self.options_panel.show_option(self.padding_option)
        else:
            self.options_panel.hide_option(self.padding_option)
            self.options_panel.show_option(self.radius_option)

class DefineCentroidDialog:
    def __init__(self, sm_tool):
        self.tool_window = tw = sm_tool.tool_window.create_child_window("Define Centroid",
            close_destroys=False)
        self.session = sm_tool.session
        layout = QVBoxLayout()
        layout.setSpacing(0)
        tw.ui_area.setLayout(layout)

        layout.addWidget(QLabel("Create centroid for selected atoms..."), alignment=Qt.AlignCenter)
        self.options_panel = options_panel = OptionsPanel(sorting=False, scrolled=False)
        layout.addWidget(options_panel)
        self.name_option = StringOption("Centroid name", "centroid", None)
        options_panel.add_option(self.name_option)
        self.color_option = ColorWithDefaultOption("Color", None, None, has_alpha_channel=True)
        options_panel.add_option(self.color_option)
        self.radius_option = AngstromOption("Radius", 2.0, None, decimal_places=1, step=0.5, min="positive")
        options_panel.add_option(self.radius_option)
        self.weighting_option = BooleanOption("Mass weighting", True, None)
        options_panel.add_option(self.weighting_option)

        bbox = qbbox(qbbox.Ok | qbbox.Apply | qbbox.Close | qbbox.Help)
        bbox.accepted.connect(self.define_centroid)
        bbox.rejected.connect(lambda tw=tw: setattr(tw, 'shown', False))
        bbox.button(qbbox.Apply).clicked.connect(lambda *args: self.define_centroid(hide=False))
        bbox.button(qbbox.Help).setEnabled(False)
        layout.addWidget(bbox)

        tw.manage(None)

    def define_centroid(self, hide=True):
        # gather parameters before hiding, so that dialog stays up if there's an error in parameters
        from chimerax.atomic import selected_atoms
        sel_atoms = selected_atoms(self.session)
        if len(sel_atoms) < 1:
            raise UserError("Need to select at least one atom")
        cmd = "define centroid sel"
        centroid_name = self.name_option.value
        if centroid_name and centroid_name != "centroid":
            cmd += " name %s" % StringArg.unparse(centroid_name)
        color = self.color_option.value
        if color is not None:
            from chimerax.core.colors import color_name
            cmd += " color " + StringArg.unparse(color_name(color))
        radius = self.radius_option.value
        if radius != 2.0:
            cmd += " radius %g" % radius
        if hide:
            self.tool_window.shown = False
        # run command here
        run(self.session, cmd)

    def _enclosed_changed(self, opt):
        if opt.value:
            self.options_panel.hide_option(self.radius_option)
            self.options_panel.show_option(self.padding_option)
        else:
            self.options_panel.hide_option(self.padding_option)
            self.options_panel.show_option(self.radius_option)
