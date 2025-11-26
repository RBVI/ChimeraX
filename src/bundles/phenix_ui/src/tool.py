# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.tools import ToolInstance
from chimerax.core.errors import UserError
from chimerax.core.settings import Settings
from Qt.QtCore import Qt
from Qt.QtWidgets import QDialog

from chimerax.ui.widgets import Citation
class PhenixCitation(Citation):
    def __init__(self, session, tool_name, phenix_name, *, title=None, info=None, **kw):
        # if title is None, use generic Phenix citation
        if title is None:
            title = \
                "Macromolecular structure determination using X-rays,<br>" \
                "neutrons and electrons: recent developments in Phenix"
            #info = [
            #    "Liebschner D, Afonine PV, Baker ML, Bunkóczi G, Chen VB,",
            #    "Croll TI, Hintze B, Hung LW, Jain S, McCoy AJ, Moriarty NW,",
            #    "Oeffner RD, Poon BK, Prisant MG, Read RJ, Richardson JS,",
            #    "Richardson DC, Sammito MD, Sobolev OV, Stockwell DH,",
            #    "Terwilliger TC, Urzhumtsev AG, Videau LL, Williams CJ,",
            #    "Adams PD",
            info = ["Liebschner D, Afonine PV, Baker ML, <i>et al.<i>"]
            info += [
                "Acta Cryst. D75, 861-877 (2019)"
            ]
            kw['pubmed_id'] = 31588918
        elif info is None:
            raise AssertionError("Citation 'title' argument supplied, but not 'info' argument")
        cite = '<br>'.join(["<b>" + title + "</b>"] + info)
        kw['prefix'] = "%s uses the Phenix <i>%s</i> command. Please cite:" % (tool_name, phenix_name)
        super().__init__(session, cite, **kw)

class DouseSettings(Settings):
    AUTO_SAVE = {
        "show_hbonds": True,
    }

from chimerax.check_waters.tool import CheckWaterViewer, check_overlap
class DouseResultsViewer(CheckWaterViewer):

    help = "help:user/tools/waterplacement.html#waterlist"

    def __init__(self, session, tool_name, orig_model=None, douse_model=None, compared_waters=None,
            map=None):
        # if 'model' is None, we are being restored from a session and _finalize_init() will be called later
        super().__init__(session, tool_name, douse_model, compare_info=(orig_model, compared_waters),
            model_labels=("input", "douse"), compare_map=map, category_tips=(
            "Waters placed by douse that are not in input model",
            "Waters in input model and also found by douse",
            "Waters in input model only"
            ))

from .douse import command_defaults as douse_defaults
class LaunchDouseSettings(Settings):
    AUTO_SAVE = {
        'first_shell': not douse_defaults['far_water'],
        'keep_waters': douse_defaults['keep_input_water'],
        'hide_map': douse_defaults['map_range'] > 0,
        'hide_map_dist': douse_defaults['map_range'],
        'res_range': douse_defaults['residue_range'],
        'verbose': douse_defaults['verbose'],
    }

class LaunchDouseTool(ToolInstance):
    help = "help:user/tools/waterplacement.html"

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name)
        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area

        from Qt.QtWidgets import QVBoxLayout, QGridLayout, QLabel
        from Qt.QtCore import Qt
        layout = QVBoxLayout()
        parent.setLayout(layout)
        #layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(1)

        input_layout = QGridLayout()
        #input_layout.setContentsMargins(0,0,0,0)
        input_layout.setSpacing(1)
        layout.addLayout(input_layout)
        input_layout.addWidget(QLabel("Place waters for structure..."), 0, 0,
            alignment=Qt.AlignHCenter|Qt.AlignBottom)
        input_layout.addWidget(QLabel("Guided by map..."), 0, 1, alignment=Qt.AlignHCenter|Qt.AlignBottom)
        from chimerax.atomic.widgets import AtomicStructureMenuButton
        self.structure_menu = AtomicStructureMenuButton(session)
        input_layout.addWidget(self.structure_menu, 1, 0, alignment=Qt.AlignHCenter|Qt.AlignTop)
        from chimerax.ui.widgets import ModelMenuButton
        from chimerax.map import Volume
        self.map_menu = ModelMenuButton(session, class_filter=Volume)
        input_layout.addWidget(self.map_menu, 1, 1, alignment=Qt.AlignHCenter|Qt.AlignTop)

        if not hasattr(self.__class__, 'settings'):
            self.__class__.settings = LaunchDouseSettings(session, "launch douse")
        from chimerax.ui.options import OptionsPanel, BooleanOption, FloatOption
        options = OptionsPanel(sorting=False, scrolled=False)
        layout.addWidget(options)
        self.keep_waters_option = BooleanOption("Keep waters from input structure",
            None, None, attr_name="keep_waters", settings=self.settings)
        options.add_option(self.keep_waters_option)
        self.first_shell_option = BooleanOption("First shell only", None, None,
            attr_name="first_shell", settings=self.settings,
            balloon="Only place waters that interact directly with structure, rather than other waters")
        options.add_option(self.first_shell_option)
        from .douse import douse_needs_resolution
        if douse_needs_resolution(session):
            self.resolution_option = FloatOption("Map resolution", 3.0, None, decimal_places=1,
                min=0.0, max=99.9, right_text="\N{ANGSTROM SIGN}")
            options.add_option(self.resolution_option)
        self.hide_map_option = BooleanOption("Hide map far from waters", None, None,
            attr_name="hide_map", settings=self.settings)
        options.add_option(self.hide_map_option)
        self.hide_map_dist_option = FloatOption("Map-hiding distance", None, None,
            attr_name="hide_map_dist", settings=self.settings, decimal_places=1, step=1, min=0, max=99,
            right_text="\N{ANGSTROM SIGN}")
        options.add_option(self.hide_map_dist_option)
        self.res_range_option = FloatOption("Show residues with atoms within this distance of waters", None,
            None, attr_name="res_range", settings=self.settings, decimal_places=1, step=1, min=0, max=99,
            right_text="\N{ANGSTROM SIGN}")
        options.add_option(self.res_range_option)
        self.verbose_option = BooleanOption("Show full douse output in log", None, None,
            attr_name="verbose", settings=self.settings)
        options.add_option(self.verbose_option)

        layout.addWidget(PhenixCitation(session, tool_name, "douse"), alignment=Qt.AlignCenter)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        self.bbox = bbox = qbbox(qbbox.Ok | qbbox.Close | qbbox.Help)
        bbox.accepted.connect(self.launch_douse)
        bbox.rejected.connect(self.delete)
        from chimerax.core.commands import run
        bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help " + self.help))
        layout.addWidget(bbox)

        tw.manage(placement=None)

    def launch_douse(self):
        structure = self.structure_menu.value
        if not structure:
            raise UserError("Must specify a structure for water placement")
        map = self.map_menu.value
        if not map:
            raise UserError("Must specify a map for water placement")
        check_overlap(structure, map)
        cmd = "phenix douse %s near %s" % (map.atomspec, structure.atomspec)
        from chimerax.core.commands import BoolArg
        first_shell = self.first_shell_option.value
        if first_shell != (not douse_defaults['far_water']):
            cmd +=  " farWater %s" % BoolArg.unparse(not first_shell)
        keep_waters = self.keep_waters_option.value
        if keep_waters != douse_defaults['keep_input_water']:
            cmd += " keepInputWater %s" % BoolArg.unparse(keep_waters)
        hide_map = self.hide_map_option.value
        if hide_map:
            hide_map_dist = self.hide_map_dist_option.value
            if hide_map_dist != douse_defaults['map_range']:
                cmd += " mapRange %g" % hide_map_dist
        elif douse_defaults['map_range'] > 0:
            cmd += " mapRange 0"
        res_range = self.res_range_option.value
        if res_range != douse_defaults['residue_range']:
            cmd += " residueRange %g" % res_range
        verbose = self.verbose_option.value
        if verbose != douse_defaults['verbose']:
            cmd += " verbose %s" % BoolArg.unparse(verbose)
        if hasattr(self, 'resolution_option'):
            cmd += " resolution %g" % self.resolution_option.value
        from chimerax.core.commands import run
        run(self.session, cmd)
        self.delete()

class EmplaceLocalResultsViewer(ToolInstance):
    help = None
    def __init__(self, session, *args):
        # if 'args' is empty, we are being restored from a session and _finalize_init() will be called later
        super().__init__(session, "Local EM Fitting Results")
        if not args:
            return
        self._finalize_init(*args)

    def _finalize_init(self, orig_model, transforms, llgs, ccs, show_sharpened_map, map_group, sym_map,
            *, table_state=None):
        self.orig_model = orig_model
        self.orig_position = orig_model.position
        self.transforms = transforms
        self.llgs = llgs
        self.ccs = ccs
        self.map_group = map_group
        self.sym_map = sym_map

        from chimerax.core.models import REMOVE_MODELS
        self._finalizing_symmetry = False
        self.handlers = [
            self.session.triggers.add_handler(REMOVE_MODELS, self._models_removed_cb),
        ]
        self._interpolate_handler = None

        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self, close_destroys=False)
        parent = tw.ui_area

        from Qt.QtWidgets import QHBoxLayout, QButtonGroup, QVBoxLayout, QRadioButton, QCheckBox
        from Qt.QtWidgets import QPushButton, QLabel, QToolButton, QGridLayout, QWidget, QDockWidget

        layout = QVBoxLayout()
        layout.setContentsMargins(2,2,2,2)
        layout.setSpacing(0)
        parent.setLayout(layout)

        # Building the table is going to call _new_selection, so the "sharpened" check box needs to exist
        # before building the table, but don't add them to the layout until after the table
        check_box_area = QWidget()
        cb_layout = QHBoxLayout()
        check_box_area.setLayout(cb_layout)
        self.sharpened_checkbox = QCheckBox("Show sharpened map")
        self.sharpened_checkbox.setChecked(show_sharpened_map)
        self.sharpened_checkbox.toggled.connect(self._show_sharpened_cb)
        cb_layout.addWidget(self.sharpened_checkbox, alignment=Qt.AlignCenter)
        if sym_map:
            self.symmetry_checkbox = QCheckBox("Show symmetry copies")
            self.symmetry_checkbox.setChecked(True)
            self.symmetry_checkbox.toggled.connect(self._show_symmetry_cb)
            cb_layout.addWidget(self.symmetry_checkbox, alignment=Qt.AlignCenter)

        self.table = self._build_table(table_state)
        layout.addWidget(self.table, stretch=1)

        layout.addWidget(check_box_area, alignment=Qt.AlignCenter)

        from chimerax.ui import shrink_font
        instructions = QLabel("Click OK to retain the chosen fit and remove others")
        shrink_font(instructions, 0.85)
        layout.addWidget(instructions, alignment=Qt.AlignCenter)

        if sym_map:
            self._show_symmetry_cb(True)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        self.bbox = bbox = qbbox(qbbox.Ok | qbbox.Close | qbbox.Help)
        if hasattr(self, 'accept_fit'):
            bbox.accepted.connect(self.accept_fit)
        else:
            bbox.button(qbbox.Ok).setEnabled(False)
        bbox.rejected.connect(self.delete)
        if self.help:
            from chimerax.core.commands import run
            bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help " + self.help))
        else:
            bbox.button(qbbox.Help).setEnabled(False)
        layout.addWidget(bbox)

        self.tool_window.manage("side")

    def accept_fit(self):
        # running commands directly in delete() can be hazardous, so...
        if self.sym_map and self.symmetry_checkbox.isChecked():
            self._finalizing_symmetry = True
            from chimerax.core.commands import run, concise_model_spec, StringArg
            modelspec = self.orig_model.atomspec
            prev_models = set(self.session.models[:])
            run(self.session,
                f"sym clear {modelspec} ; sym {modelspec} symmetry {self.sym_map.atomspec} copies true")
            added = [m for m in self.session.models if m not in prev_models]
            orig_id, orig_name = self.orig_model.id[0], self.orig_model.name
            run(self.session, f"close {modelspec}")
            run(self.session, "combine " + concise_model_spec(self.session, added) + " close true"
                " modelId %d name %s" % (orig_id, StringArg.unparse(orig_name)))
        self.delete()

    def delete(self):
        for handler in self.handlers:
            handler.remove()
        if self._interpolate_handler:
            self._interpolate_handler.remove()
        if not self.map_group.deleted:
            close_group = True
            if self.sharpened_checkbox.isChecked():
                sel = self.table.selected
                for datum in self.table.data:
                    if datum in sel:
                        close_group = False
                    else:
                        smap = self._sharpened_map(datum.num)
                        if smap:
                            self.session.models.close([smap])
            if close_group:
                self.session.models.close([self.map_group])
        self.map_group = self.orig_model = self.sym_map = None
        super().delete()

    def _build_table(self, table_state):
        class TableDatum:
            def __init__(self, num, transform, llg, cc):
                self.num = num
                self.transform = transform
                self.llg = llg
                self.cc = cc
        from chimerax.ui.widgets import ItemTable
        table = ItemTable()
        result_col = table.add_column("Result", "num")
        table.add_column("Correlation Coefficient", "cc", format="%.3g")
        table.add_column("Log-Likelihood Gain", "llg", format="%.3g")
        table.data = [TableDatum(*args)
            for args in zip(range(1, len(self.transforms)+1), self.transforms, self.llgs, self.ccs)]
        table.launch(select_mode=table.SelectionMode.SingleSelection, session_info=table_state)
        table.sort_by(result_col, table.SORT_ASCENDING)
        table.selection_changed.connect(self._new_selection)
        table.selected = table.data[0:1]
        return table

    def _interpolate(self, destination):
        if self._interpolate_handler:
            self._interpolate_handler.remove()
        # interpolation code largely cribbed from map_fit.search.move_models/move_step
        def make_step(trig_name, trig_data, *, tool=self, destination=destination, frame_info=[10]):
            m = tool.orig_model
            b = m.bounds()
            finish = False
            if b:
                num_frames = frame_info[0]
                cscene = .5 * (b.xyz_min + b.xyz_max)
                c = m.scene_position.inverse() * cscene # Center in model coordinates
                m.position = m.position.interpolate(destination, c, 1.0/num_frames)
                if num_frames > 1:
                    frame_info[0] = num_frames - 1
                else:
                    finish = True
            else:
                m.position = destination
                finish = True
            if finish:
                tool._interpolate_handler.remove()
                tool._interpolate_handler = None
                if self.sym_map and self.symmetry_checkbox.isChecked():
                    from chimerax.core.commands import run
                    run(self.session, "sym " + self.orig_model.atomspec + " symmetry "
                        + self.sym_map.atomspec + " copies false", log=False)
        self._interpolate_handler = self.session.triggers.add_handler("new frame", make_step)

    def _models_removed_cb(self, trig_name, trig_data):
        if self.orig_model in trig_data and not self._finalizing_symmetry:
            self.delete()

    def _new_selection(self, selected, unselected):
        if not selected:
            return
        from chimerax.geometry import Place
        destination = Place(selected[0].transform) * self.orig_position
        if unselected:
            self._interpolate(destination)
        else:
            self.orig_model.position = destination

        if self.sharpened_checkbox.isChecked():
            for data, disp_state in [(unselected, False), (selected, True)]:
                for datum in data:
                    smap = self._sharpened_map(datum.num)
                    if smap:
                        smap.display = disp_state

    def _sharpened_map(self, num):
        if self.map_group.deleted:
            return None
        for child in self.map_group.child_models():
            if child.name == "map %d" % num:
                return child
        return None

    def _show_sharpened_cb(self, checked):
        for datum in self.table.selected:
            smap = self._sharpened_map(datum.num)
            if smap:
                smap.display = checked

    def _show_symmetry_cb(self, checked):
        from chimerax.core.commands import run
        if checked:
            run(self.session, "sym " + self.orig_model.atomspec + " symmetry " + self.sym_map.atomspec
                + " copies false")
        else:
            run(self.session, "sym clear " + self.orig_model.atomspec)

class LaunchEmplaceLocalTool(ToolInstance):
    help = "help:user/tools/localemfitting.html"

    CENTER_MODEL = "center of model..."
    CENTER_SELECTION = "center of selection"
    CENTER_VIEW = "center of view"
    CENTER_XYZ = "specified xyz position..."
    CENTERING_METHODS = [CENTER_MODEL, CENTER_SELECTION, CENTER_VIEW, CENTER_XYZ]

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name)
        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self, close_destroys=False)
        parent = tw.ui_area

        if not hasattr(self.__class__, 'settings'):
            self.__class__.settings = LaunchEmplaceLocalSettings(session, "launch emplace local")

        from Qt.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QWidget, QPushButton, QMenu, QLineEdit
        from Qt.QtWidgets import QCheckBox, QGridLayout, QGroupBox
        from Qt.QtGui import QDoubleValidator
        from Qt.QtCore import Qt
        layout = QVBoxLayout()
        parent.setLayout(layout)
        #layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(1)

        centering_widget = QWidget()
        layout.addWidget(centering_widget, alignment=Qt.AlignCenter, stretch=1)
        structure_layout = QGridLayout()
        structure_layout.setSpacing(1)
        centering_widget.setLayout(structure_layout)
        structure_layout.addWidget(QLabel("Fit "), 0, 0, alignment=Qt.AlignRight)
        from chimerax.atomic.widgets import AtomicStructureMenuButton, AtomicStructureListWidget
        self.structure_menu = AtomicStructureMenuButton(session)
        structure_layout.addWidget(self.structure_menu, 0, 1)
        structure_layout.addWidget(QLabel(" using "), 0, 2)
        self.HALF_MAPS, self.FULL_MAP = self.menu_items = ["half maps", "full map"]
        self.mt_explanations = {
            self.HALF_MAPS: "Using half maps is recommended if available.  If local resolution is not specified (i.e. is zero) then it will be estimated from the half maps.",
            self.FULL_MAP: "Using only the full map is less reliable than using half maps, so use half maps if available.  If using a full map, then specifying a (non-zero) resolution is mandatory."
        }
        assert len(self.menu_items) == len(self.mt_explanations)
        self.map_type_mb = QPushButton(self.HALF_MAPS)
        structure_layout.addWidget(self.map_type_mb, 0, 3)
        mt_menu = QMenu(self.map_type_mb)
        mt_menu.triggered.connect(self._map_type_changed)
        for item in self.menu_items:
            mt_menu.addAction(item)
        self.map_type_mb.setMenu(mt_menu)
        from chimerax.ui.widgets import ModelListWidget, ModelMenuButton
        ex_lab = self.mt_explanation_label = QLabel(self.mt_explanations[self.HALF_MAPS])
        ex_lab.setWordWrap(True)
        ex_lab.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        from chimerax.ui import shrink_font
        shrink_font(ex_lab, fraction=0.85)
        structure_layout.addWidget(ex_lab, 1, 0, 1, 4)
        class ShortMLWidget(ModelListWidget):
            def sizeHint(self):
                hint = super().sizeHint()
                hint.setHeight(hint.height()//2)
                return hint
        from chimerax.map import Volume
        self.half_map_list = ShortMLWidget(session, class_filter=Volume)
        structure_layout.addWidget(self.half_map_list, 0, 4, 2, 1, alignment=Qt.AlignLeft)
        structure_layout.setRowStretch(1, 1)
        structure_layout.setColumnStretch(4, 1)

        from chimerax.ui.options import OptionsPanel, FloatOption
        res_options = OptionsPanel(scrolled=False, contents_margins=(0,0,0,0))
        layout.addWidget(res_options, alignment=Qt.AlignCenter)
        self.res_option = FloatOption("Local map resolution:", None, None, min=0.0, decimal_places=2,
            step=0.1, max=99.99, balloon="Map resolution in the search region.\n"
            "If unknown, and using half maps, providing a value of zero will cause an estimated\n"
            "resolution to be used.  For full maps, providing the resolution is mandatory.")
        res_options.add_option(self.res_option)

        prefitted_widget = QWidget()
        layout.addWidget(prefitted_widget, alignment=Qt.AlignCenter, stretch=1)
        prefitted_layout = QHBoxLayout()
        prefitted_layout.setSpacing(1)
        prefitted_widget.setLayout(prefitted_layout)
        label_container = QWidget()
        prefitted_layout.addWidget(label_container, alignment=Qt.AlignRight)
        label_layout = QVBoxLayout()
        label_layout.setSpacing(0)
        label_layout.setContentsMargins(0,0,0,0)
        label_container.setLayout(label_layout)
        prefitted_tip = '''If any structures have already been fit into
other parts of the map, specify those here.'''
        prefitted_label = QLabel("Pre-fitted structures (if any):")
        prefitted_label.setToolTip(prefitted_tip)
        label_layout.addWidget(prefitted_label, alignment=Qt.AlignBottom|Qt.AlignHCenter)
        warning_label = QLabel("Requires Phenix 2.0 or later")
        shrink_font(warning_label)
        label_layout.addWidget(warning_label, alignment=Qt.AlignTop|Qt.AlignHCenter)
        class ShortASLWidget(AtomicStructureListWidget):
            def sizeHint(self):
                hint = super().sizeHint()
                hint.setHeight(hint.height()//2)
                return hint
        self.prefitted_list = ShortASLWidget(session, autoselect=ShortASLWidget.AUTOSELECT_NONE,
            filter_func=lambda s, *args, sm=self.structure_menu: s != sm.value)
        self.structure_menu.value_changed.connect(self.prefitted_list.refresh)
        prefitted_layout.addWidget(self.prefitted_list, alignment=Qt.AlignLeft)

        centering_widget = QWidget()
        layout.addWidget(centering_widget, alignment=Qt.AlignCenter, stretch=1)
        centering_layout = QHBoxLayout()
        centering_layout.setSpacing(1)
        centering_widget.setLayout(centering_layout)
        centering_tip = '''How to specify the center of the fitting search.  Choices are:

%s — If the center of rotation is being displayed ("cofr showPivot true") use that.  Otherwise,
    if the center of rotation is a fixed point ("cofr fixed") use that.  If neither of those is
    true, use the midpoint of where the center of the window intersects the front and back of
    the bounding box of the map.

%s — The center of a particular model, frequently the map, or the structure to be fitted once
    it has been approximately positioned.

%s - The center of the bounding box enclosing currently selected objects.

%s — A specific X/Y/Z position, given in angstroms relative to the origin of the map.
        ''' % (self.CENTER_VIEW.rstrip('.'), self.CENTER_MODEL.rstrip('.'),
            self.CENTER_SELECTION.rstrip('.'), self.CENTER_XYZ.rstrip('.'))
        centering_label = QLabel("Center search at")
        centering_label.setToolTip(centering_tip)
        centering_layout.addWidget(centering_label, alignment=Qt.AlignRight)
        self.centering_button = QPushButton()
        self.centering_button.setToolTip(centering_tip)
        centering_layout.addWidget(self.centering_button)
        centering_menu = QMenu(self.centering_button)
        for method in self.CENTERING_METHODS:
            centering_menu.addAction(method)
        centering_menu.triggered.connect(lambda act: self._set_centering_method(act.text()))
        self.centering_button.setMenu(centering_menu)
        self.xyz_area = QWidget()
        xyz_layout = QHBoxLayout()
        xyz_layout.setSpacing(1)
        self.xyz_area.setLayout(xyz_layout)
        self.xyz_widgets = []
        for lab in ["X", " Y", " Z"]:
            xyz_layout.addWidget(QLabel(lab), alignment=Qt.AlignRight)
            entry = QLineEdit()
            entry.setValidator(QDoubleValidator())
            entry.setAlignment(Qt.AlignCenter)
            entry.setMaximumWidth(50)
            entry.setText("0")
            xyz_layout.addWidget(entry, alignment=Qt.AlignLeft)
            self.xyz_widgets.append(entry)
        self.model_menu = ModelMenuButton(session)
        centering_layout.addWidget(self.model_menu)
        centering_layout.addWidget(self.xyz_area)
        self._set_centering_method()

        checkbox_area = QWidget()
        layout.addWidget(checkbox_area, alignment=Qt.AlignCenter)
        checkbox_layout = QVBoxLayout()
        checkbox_layout.setContentsMargins(0,0,0,0)
        checkbox_area.setLayout(checkbox_layout)
        self.verify_center_checkbox = QCheckBox("Interactively verify/adjust center before searching")
        self.verify_center_checkbox.setChecked(True)
        checkbox_layout.addWidget(self.verify_center_checkbox, alignment=Qt.AlignLeft)
        self.opaque_maps_checkbox = QCheckBox("Make maps opaque while verifying center")
        self.opaque_maps_checkbox.setToolTip(
            "ChimeraX cannot show multiple transparent objects correctly, so make maps opaque\n"
            "while transparent interactive search-center sphere is being displayed"
        )
        self.opaque_maps_checkbox.setChecked(self.settings.opaque_maps)
        self.verify_center_checkbox.clicked.connect(lambda checked, b=self.opaque_maps_checkbox:
            b.setHidden(not checked))
        checkbox_layout.addWidget(self.opaque_maps_checkbox, alignment=Qt.AlignLeft)
        self.show_sharpened_map_checkbox = QCheckBox("Show locally sharpened map computed by Phenix")
        self.show_sharpened_map_checkbox.setChecked(self.settings.show_sharpened_map)
        self.show_sharpened_map_checkbox.setToolTip(
            "Phenix.emplace_local computes a sharpened map for the region being searched.\n"
            "This sharpened map is what is actually used for the fitting.  This checkbox\n"
            "controls whether the sharpened map is initially shown in ChimeraX once the\n"
            "calculation completes.  Even if not checked, the map will be opened (but hidden)."
        )
        checkbox_layout.addWidget(self.show_sharpened_map_checkbox, alignment=Qt.AlignLeft)
        self.symmetry_checkbox = QCheckBox("After fitting, add symmetry copies"
            " automatically if map symmetry is detected")
        checkbox_layout.addWidget(self.symmetry_checkbox, alignment=Qt.AlignLeft)
        layout.addSpacing(10)

        layout.addWidget(PhenixCitation(session, tool_name, "emplace_local",
            title="Likelihood-based interactive local docking into cryo-EM maps in ChimeraX",
            info=["Read RJ, Pettersen EF, McCoy AJ, Croll TI, Terwilliger TC, Poon BK, Meng EC",
                "Liebschner D, Adams PD", "Acta Cryst. D80, 588-598 (2024)"],
            pubmed_id=39058381), alignment=Qt.AlignCenter)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        self.bbox = bbox = qbbox(qbbox.Ok | qbbox.Apply | qbbox.Close | qbbox.Help)
        bbox.accepted.connect(self.launch_emplace_local)
        bbox.button(qbbox.Apply).clicked.connect(lambda *args: self.launch_emplace_local(apply=True))
        bbox.rejected.connect(self.delete)
        if self.help:
            from chimerax.core.commands import run
            bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help " + self.help))
        else:
            bbox.button(qbbox.Help).setEnabled(False)
        layout.addWidget(bbox)

        tw.manage(placement=None)

    def launch_emplace_local(self, apply=False):
        structure = self.structure_menu.value
        if not structure:
            raise UserError("Must specify a structure to fit")
        map_type = self.map_type_mb.text()
        maps = self.half_map_list.value
        res = self.res_option.value
        if map_type == self.HALF_MAPS:
            if len(maps) != 2:
                raise UserError("Must specify exactly two half maps for fitting")
        else:
            if len(maps) != 1:
                raise UserError("Must specify exactly one full map for fitting")
            if res == 0.0:
                raise UserError("Must specify a resolution value for the full map")
        prefitted = self.prefitted_list.value
        method = self.centering_button.text()
        if method == self.CENTER_XYZ:
            center = [float(widget.text()) for widget in self.xyz_widgets]
        elif method == self.CENTER_MODEL:
            centering_model = self.model_menu.value
            if centering_model is None:
                raise UserError("No model chosen for specifying search center")
            bnds = centering_model.bounds()
            if bnds is None:
                raise UserError("No part of model for specifying search center is displayed")
            center = bnds.center()
        elif method == self.CENTER_VIEW:
            # If pivot point shown or using fixed center of rotation, use that.
            # Otherwise, midpoint where center of window intersects front and back of halfmap bounding box.
            view_center = None
            mv = self.session.main_view
            for d in mv.drawing.child_drawings():
                if d.__class__.__name__ == "PivotIndicator":
                    view_center = d.position.origin()
                    break
            else:
                if mv.center_of_rotation_method == "fixed":
                    view_center = mv.center_of_rotation
            if view_center is None:
                from chimerax.map import Volume
                shown_vols = [v for v in self.session.models if isinstance(v, Volume) and v.display]
                if len(shown_vols) == 1:
                    view_map = shown_vols[0]
                else:
                    view_map = maps[0]
                from .emplace_local import view_box, ViewBoxError
                try:
                    view_center = view_box(self.session, view_map)
                except ViewBoxError as e:
                    raise UserError(str(e))
            center = view_center
        elif method == self.CENTER_SELECTION:
            if self.session.selection.empty():
                raise UserError("Nothing selected")
            from chimerax.atomic import selected_atoms
            sel_atoms = selected_atoms(self.session)
            from chimerax.geometry import point_bounds, union_bounds
            atom_bbox = point_bounds(sel_atoms.scene_coords)
            atom_models = set(sel_atoms.unique_structures)
            bbox = union_bounds([atom_bbox]
                + [m.bounds() for m in self.session.selection.models() if m not in atom_models])
            if bbox is None:
                raise UserError("No bounding box for selected items")
            center = bbox.center()
        else:
            raise AssertionError("Unknown centering method")
        self.settings.search_center = method
        self.settings.show_sharpened_map = ssm = self.show_sharpened_map_checkbox.isChecked()
        apply_symmetry = self.symmetry_checkbox.isChecked()
        if self.verify_center_checkbox.isChecked():
            self.settings.opaque_maps = self.opaque_maps_checkbox.isChecked()
            VerifyELCenterDialog(self.session, structure, maps, res, prefitted, center,
                self.settings.opaque_maps, ssm, apply_symmetry)
        else:
            _run_emplace_local_command(self.session, structure, maps, res, prefitted, center,
                ssm, apply_symmetry)
        if not apply:
            self.display(False)

    def _map_type_changed(self, action):
        self.map_type_mb.setText(action.text())
        self.mt_explanation_label.setText(self.mt_explanations[action.text()])

    def _set_centering_method(self, method=None):
        if method is None:
            method = self.settings.search_center
            if method not in self.CENTERING_METHODS:
                method = self.CENTER_MODEL
        self.centering_button.setText(method)
        self.xyz_area.setHidden(True)
        self.model_menu.setHidden(True)
        if method == self.CENTER_XYZ:
            self.xyz_area.setHidden(False)
        elif method == self.CENTER_MODEL:
            self.model_menu.setHidden(False)

class LaunchEmplaceLocalSettings(Settings):
    AUTO_SAVE = {
        'search_center': LaunchEmplaceLocalTool.CENTER_MODEL,
        'opaque_maps': True,
        'show_sharpened_map': False
    }

class LaunchFitLoopsTool(ToolInstance):
    help = "help:user/tools/fitloops.html"

    GAPS_TAB, REMODEL_TAB = TAB_NAMES = ("Fill Gaps", "Remodel")

    ADVISORY_RES_LIMIT = 15
    HARD_RES_LIMIT = 20

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name)
        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self, close_destroys=False)
        parent = tw.ui_area

        #if not hasattr(self.__class__, 'settings'):
        #    self.__class__.settings = LaunchEmplaceLocalSettings(session, "launch emplace local")

        from Qt.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QWidget, QSpinBox, QStackedWidget
        from Qt.QtWidgets import QGridLayout, QAbstractItemView
        from Qt.QtCore import Qt, QSize
        layout = QVBoxLayout()
        parent.setLayout(layout)
        parent.setMinimumSize(0, 400)
        #layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(1)

        centering_widget = QWidget()
        layout.addWidget(centering_widget, alignment=Qt.AlignCenter)
        data_layout = QHBoxLayout()
        data_layout.setSpacing(1)
        data_layout.setContentsMargins(0,0,0,0)
        centering_widget.setLayout(data_layout)
        data_layout.addWidget(QLabel("Structure: "), alignment=Qt.AlignRight)
        from chimerax.atomic.widgets import AtomicStructureMenuButton
        self.structure_menu = AtomicStructureMenuButton(session)
        self.structure_menu.value_changed.connect(self._input_changed)
        data_layout.addWidget(self.structure_menu, alignment=Qt.AlignLeft)
        data_layout.setStretch(data_layout.count(), 1)
        data_layout.addWidget(QLabel("  Map: "), alignment=Qt.AlignRight)
        from chimerax.ui.widgets import ModelMenuButton
        from chimerax.map import Volume
        self.map_menu = ModelMenuButton(session, class_filter=Volume)
        self.map_menu.value_changed.connect(self._input_changed)
        data_layout.addWidget(self.map_menu, alignment=Qt.AlignLeft, stretch=1)

        self.target_area = QStackedWidget()

        layout.addWidget(self.target_area, stretch=1, alignment=Qt.AlignCenter)
        self.need_input_message = "Select a structure and map from the menus above"
        self.no_table_label = QLabel(self.need_input_message)
        self.no_table_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.no_table_label.setWordWrap(True)
        self.no_table_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        from chimerax.core.commands import run
        self.no_table_label.linkActivated.connect(lambda link_text, ses=session, run=run:
            run(ses, "help help:" + link_text))
        self.target_area.addWidget(self.no_table_label)

        self.table_area = QWidget()
        self.table_area.setMinimumHeight(300)
        table_layout = QVBoxLayout()
        table_layout.setSpacing(2)
        self.table_area.setLayout(table_layout)
        self.target_area.addWidget(self.table_area)

        self.help_label = QLabel()
        self.help_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        from chimerax.core.commands import run
        self.help_label.linkActivated.connect(lambda *args, ses=session, run=run:
            run(ses, "help help:user/selection.html"))
        self.model_structure_message = '&bull; Specify the parts of %%s to model into %%s' \
            ' by <a href="help:select">selection</a> using' \
            ' <a href="help:select">any method</a>,' \
            ' including the table below.<br>' \
            '&bull; Modeling >= %d consecutive residues is <span style="color:rgb(219, 118, 0)">not recommended</span>, and' \
            ' modeling >= %d consecutive residues is <span style="color:red">disallowed</span>.' % (
            self.ADVISORY_RES_LIMIT, self.HARD_RES_LIMIT)
        self.help_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.help_label.setWordWrap(True)
        from chimerax.ui import shrink_font
        shrink_font(self.help_label, 0.9)
        table_layout.addWidget(self.help_label)
        table_layout.addWidget(QLabel("Gaps"), alignment=Qt.AlignHCenter|Qt.AlignBottom)
        from chimerax.ui.widgets import ItemTable
        self.target_table = targets = ItemTable(session=session)
        chain_col = targets.add_column("Chain", "chain_id")
        targets.add_column("Adjacent Residues", "between")
        self.gap_column = targets.add_column("Gap Length", "length", data_color=self._gap_color)
        targets.selection_changed.connect(self._new_target)
        targets.launch(select_mode=QAbstractItemView.SelectionMode.SingleSelection)
        targets.sort_by(chain_col, targets.SORT_ASCENDING)
        table_layout.addWidget(targets, alignment=Qt.AlignCenter, stretch=1)
        padding_area = QWidget()
        padding_layout = QHBoxLayout()
        padding_layout.setSpacing(1)
        padding_area.setLayout(padding_layout)
        table_layout.addWidget(padding_area, alignment=Qt.AlignHCenter|Qt.AlignTop)
        padding_layout.addWidget(QLabel("Also select "), alignment=Qt.AlignRight)
        self.padding_widget = QSpinBox()
        self.padding_widget.setValue(1)
        self.padding_widget.setRange(0, 99)
        self.padding_widget.valueChanged.connect(self._padding_changed)
        padding_layout.addWidget(self.padding_widget)
        self.residue_label = QLabel(" residue on each side of gap")
        padding_layout.addWidget(self.residue_label)
        self.warning_label = QLabel("")
        self.many_residues_message = "Current selection of %%d consecutive residues in chain %%s is >= %d" \
            " (fit_loops modeling not recommended)." % self.ADVISORY_RES_LIMIT
        self.too_many_residues_message = "Current selection of %%d consecutive residues in chain %%s" \
            " is >= %d (fit_loops modeling disallowed)" % self.HARD_RES_LIMIT
        self.warning_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.warning_label.setWordWrap(True)
        from chimerax.ui import shrink_font
        shrink_font(self.warning_label, 0.9)
        self.warning_label.hide()
        table_layout.addWidget(self.warning_label)
        self.target_area.setCurrentWidget(self.no_table_label)

        layout.addWidget(PhenixCitation(session, tool_name, "fit_loops"), alignment=Qt.AlignCenter)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        self.bbox = bbox = qbbox(qbbox.Ok | qbbox.Apply | qbbox.Close | qbbox.Help)
        bbox.accepted.connect(self.launch_fit_loops)
        bbox.button(qbbox.Apply).clicked.connect(lambda *args: self.launch_fit_loops(apply=True))
        bbox.rejected.connect(self.delete)
        if self.help:
            from chimerax.core.commands import run
            bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help " + self.help))
        else:
            bbox.button(qbbox.Help).setEnabled(False)
        layout.addWidget(bbox)

        from chimerax.core.selection import SELECTION_CHANGED
        self.handler = session.triggers.add_handler(SELECTION_CHANGED, self._sel_changed)

        tw.manage(placement=None)

    def delete(self):
        self.handler.remove()
        super().delete()

    def launch_fit_loops(self, apply=False):
        structure = self.structure_menu.value
        if not structure:
            raise UserError("Must specify a structure to model/remodel")
        map = self.map_menu.value
        if not map:
            raise UserError("Must specify a volume/map to guide remodeling")
        from chimerax.atomic import concise_residue_spec, Residue
        from chimerax.core.commands import run
        # generate commands for selected pseudobonds (without both endpoint atoms selected)
        # and then command for selected residues.  Execute them all simultaneously.
        commands = []
        try:
            pbs = structure.pbg_map[structure.PBG_MISSING_STRUCTURE].pseudobonds
        except KeyError:
            pbs = []
        for pb in pbs:
            if not pb.selected:
                continue
            a1, a2 = pb.atoms
            if a1.selected and a2.selected:
                continue
            r1, r2 = a1.residue, a2.residue
            if r1 == r2:
                continue
            commands.append("phenix fitLoops %s%s in %s gapOnly true"
                % (r1.atomspec, r2.atomspec, map.atomspec))

        sel_residues = structure.atoms[structure.atoms.selecteds == True].unique_residues
        sel_residues = sel_residues.filter(sel_residues.polymer_types == Residue.PT_AMINO)
        if sel_residues:
            commands.append("phenix fitLoops %s in %s"
                % (concise_residue_spec(self.session, sel_residues), map.atomspec))

        if not commands:
            raise UserError("No selected amino-acid residues or missing-structure pseudobonds in %s"
                % structure)

        run(self.session, " ; ".join(commands))
        if not apply:
            self.display(False)

    def _find_gaps(self, structure):
        try:
            pbs = structure.pbg_map[structure.PBG_MISSING_STRUCTURE].pseudobonds
        except KeyError:
            return [], []
        for chain in structure.chains:
            if chain.full_sequence_known:
                break
        else:
            return [], []
        gaps = []
        unk_gaps = []
        for pb in pbs:
            a1, a2 = pb.atoms
            r1, r2 = (a1.residue, a2.residue) if a1 < a2 else (a2.residue, a1.residue)
            if r1 == r2:
                continue
            if r1.polymer_type == r1.PT_AMINO:
                if r1.name == "UNK" or r2.name == "UNK":
                    unk_gaps.append((r1, r2, pb))
                else:
                    gaps.append((r1, r2, pb))
        gaps.sort()
        unk_gaps.sort()
        return gaps, unk_gaps

    def _gap_color(self, datum):
        full_length = datum.length + 2 * self.padding_widget.value()
        if full_length < self.ADVISORY_RES_LIMIT:
            return 'black'
        if full_length < self.HARD_RES_LIMIT:
            return [x/255 for x in (219, 118, 0)]
        return 'red'

    def _new_target(self, *args):
        table_data = self.target_table.selected
        if not table_data:
            return
        r1, r2, pb = table_data[0].gap_info
        pad = self.padding_widget.value()
        from chimerax.core.commands import run
        if pad > 0:
            index1 = r1.chain.residues.index(r1)
            index2 = r2.chain.residues.index(r2)
            bound1 = 0
            bound2 = len(r2.chain) - 1
            target_residues = []
            for offset in range(pad):
                for base_index, res_list, dir, bound in [(index1, r1.chain.residues, -1, bound1),
                        (index2, r2.chain.residues, 1, bound2)]:
                    index = base_index + dir * offset
                    if dir < 0:
                        if index < bound:
                            continue
                    elif index > bound:
                        continue
                    r = res_list[index]
                    if r:
                        target_residues.append(r)
            with self.session.undo.aggregate("Fit Loops launcher table row"):
                from chimerax.atomic import concise_residue_spec
                run(self.session, "select " + concise_residue_spec(self.session, target_residues))
                run(self.session, "view sel")
        else:
            # There is no command to select _just_ a pseudobond, so if the padding is zero...
            self.session.selection.clear()
            pb.selected = True
            a1, a2 = pb.atoms
            run(self.session, "view " + a1.atomspec + a2.atomspec)

    def _padding_changed(self, padding):
        prefix = " residue" if padding == 1 else " residues"
        cur_text = self.residue_label.text()
        self.residue_label.setText(prefix + cur_text[cur_text[1:].index(' ')+1:])
        self._new_target(self.target_table.selected)
        self.target_table.update_column(self.gap_column, data_color=True)

    def _sel_changed(self, trig_name, data):
        structure = self.structure_menu.value
        for but in self.bbox.buttons():
            if but.text() in ["OK", "Apply"]:
                but.setEnabled(True)
        structure = self.structure_menu.value
        map = self.map_menu.value
        if structure and map:
            # check if selection >= ADVISORY_RES_LIMIT consecutive chain residues
            from chimerax.atomic import Structure, selected_residues, selected_pseudobonds
            sel_res = set(selected_residues(self.session))
            for chain in structure.chains:
                num_sel = 0
                prev_existing_res = None
                gap_len = 0
                for r in chain.residues:
                    if r:
                        if gap_len > 0:
                            # Is there a selected missing-structure pseudobond between the existing residues?
                            end_points = set([r, prev_existing_res])
                            for pb in selected_pseudobonds(self.session):
                                if pb.group.name != Structure.PBG_MISSING_STRUCTURE:
                                    continue
                                if pb.atoms[0].residue in end_points and pb.atoms[1].residue in end_points:
                                    num_sel += gap_len
                                    break
                            else:
                                if num_sel >= self.ADVISORY_RES_LIMIT:
                                    break
                                num_sel = 0
                            gap_len = 0
                        if r in sel_res:
                            num_sel += 1
                        else:
                            if num_sel >= self.ADVISORY_RES_LIMIT:
                                break
                            num_sel = 0
                        prev_existing_res = r
                    else:
                        if prev_existing_res:
                            gap_len += 1
                if num_sel >= self.HARD_RES_LIMIT:
                    self.warning_label.setText(self.too_many_residues_message % (num_sel, chain.chain_id))
                    self.warning_label.setStyleSheet("QLabel { color : red }")
                    self.warning_label.show()
                    for but in self.bbox.buttons():
                        if but.text() in ["OK", "Apply"]:
                            but.setEnabled(False)
                    break
                if num_sel >= self.ADVISORY_RES_LIMIT:
                    self.warning_label.setText(self.many_residues_message % (num_sel, chain.chain_id))
                    self.warning_label.setStyleSheet("QLabel { color : rgb(219, 118, 0) }")
                    self.warning_label.show()
                    break
            else:
                self.warning_label.setText("")
                self.warning_label.hide()
            self.help_label.setText(self.model_structure_message % (structure, self.map_menu.value))
            self.target_area.setCurrentWidget(self.table_area)
        else:
            self.no_table_label.setText(self.need_input_message)
            self.target_area.setCurrentWidget(self.no_table_label)

    def _input_changed(self):
        structure = self.structure_menu.value
        map = self.map_menu.value
        if structure and map:
            if structure.chains:
                gap_info, unk_gaps = self._find_gaps(structure)
                if unk_gaps:
                    self.session.logger.info("Phenix loop fitting cannot handle gaps involving UNK residues"
                        " and therefore the following gaps have not been included in the dialog's list of"
                        " gaps:")
                    self.session.logger.info('<ul>%s</ul>\n' % ('\n'.join(
                        ['<li><a href="cxcmd:view %s%s">%s&rarr;%s</a></li>'
                        % (r1.atomspec, r2.atomspec, r1, r2) for r1, r2, pb in unk_gaps])), is_html=True)
                if gap_info:
                    msg = self.model_structure_message % (structure, self.map_menu.value)
                    class TableDatum:
                        def __init__(self, gap_info):
                            self.gap_info = gap_info
                            r1, r2, pb = gap_info
                            self.chain_id = r1.chain_id
                            self.between = "%s \N{LEFT RIGHT ARROW} %s" % (
                                r1.string(residue_only=True), r2.string(residue_only=True))
                            i1 = r1.chain.residues.index(r1)
                            i2 = r1.chain.residues.index(r2)
                            self.length = i2-i1-1
                    data = [TableDatum(gi) for gi in gap_info]
                    self.target_table.data = data
                    self.target_table.resizeColumnsToContents()
                    self.target_table.resizeRowsToContents()
                    if unk_gaps:
                        msg += f"  {structure} also has missing-structure gaps involving UNK residues," \
                            " which Phenix loop fitting cannot handle (see Log for more info)."
                    self.help_label.setText(msg)
                    self.target_area.setCurrentWidget(self.table_area)
                else:
                    self.target_table.data = []
                    if unk_gaps:
                        msg = f"{structure} only has missing-structure gaps involving UNK residues, which" \
                            " Phenix loop fitting cannot handle (see Log for more info).  You could" \
                            " remodel other residues by selecting them."
                    else:
                        for chain in structure.chains:
                            if chain.full_sequence_known:
                                seq_known = True
                                break
                        else:
                            seq_known = False
                        if seq_known:
                            msg = f"Select residues you wish to remodel."
                        else:
                            msg = f"The input file of {structure} did not include its full sequence, and" \
                                " without that information, the missing segments cannot be modeled because" \
                                " their sequences are unknown. The full sequence can be added by opening a" \
                                " <a href=\"user/commands/open.html#sequence\">file</a> containing that" \
                                " sequence or <a href=\"user/fetch.html\">fetching</a> it from UniProt," \
                                " and in the resulting Sequence window, choosing context menu entry" \
                                " Structure\N{RIGHTWARDS ARROW}Update Chain Sequence to add the sequence" \
                                " information to the associated protein structure chain (<a" \
                                " href=\"user/tools/sequenceviewer.html#context\">details...</a>)."
                    self.no_table_label.setText(msg)
                    self.target_area.setCurrentWidget(self.no_table_label)
            else:
                self.no_table_label.setText("%s has no polymeric chains!" % structure)
                self.target_area.setCurrentWidget(self.no_table_label)
        else:
            self.no_table_label.setText(self.need_input_message)
            self.target_area.setCurrentWidget(self.no_table_label)

import abc

class VerifyCenterDialog(QDialog):
    def __init__(self, session, model):
        super().__init__()
        self.session = session
        self.model = model

        from chimerax.core.models import REMOVE_MODELS
        self.vcd_handler = session.triggers.add_handler(REMOVE_MODELS, self._check_still_valid)

        from Qt.QtWidgets import QVBoxLayout, QLabel
        layout = QVBoxLayout()
        self.setLayout(layout)
        instructions = QLabel(self.instructions)
        instructions.setWordWrap(True)
        instructions.setAlignment(Qt.AlignCenter)
        layout.addWidget(instructions)

        self.add_custom_widgets(layout)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        bbox = qbbox(qbbox.Cancel)
        bbox.addButton(self.search_button_label, bbox.AcceptRole)
        bbox.accepted.connect(self.launch)
        bbox.accepted.connect(self.close)
        bbox.rejected.connect(self.close)
        layout.addWidget(bbox)

        self.show()

    def add_custom_widgets(self, layout):
        pass

    @property
    @abc.abstractmethod
    def check_models(self):
        pass

    def closeEvent(self, event):
        if not self.model.deleted:
            self.session.models.close([self.model])
        self.vcd_handler.remove()
        return super().closeEvent(event)

    @property
    @abc.abstractmethod
    def instructions(self):
        pass

    @property
    @abc.abstractmethod
    def launch(self):
        pass

    @property
    @abc.abstractmethod
    def search_button_label(self):
        pass

    def _check_still_valid(self, trig_name, removed_models):
        for rm in removed_models:
            if rm in self.check_models + [self.model]:
                self.close()
                break

class VerifyMarkerCenterDialog(VerifyCenterDialog):
    def __init__(self, session, initial_center, opaque_maps):
        self.opaque_maps = opaque_maps

        if opaque_maps:
            from chimerax.map import VolumeSurface
            self.opaque_data = {}
            for m in session.models:
                if isinstance(m, VolumeSurface) and m.rgba[-1] < 1.0:
                    self.opaque_data[m] = m.rgba[-1]
                    rgba = list(m.rgba)
                    rgba[-1] = 1.0
                    m.rgba = tuple(rgba)

        self.marker_set_id = session.models.next_id()[0]
        from chimerax.core.commands import run
        self.marker = run(session, "marker #%d position %g,%g,%g radius %g color 100,65,0,50"
            % (self.marker_set_id, *initial_center, self.search_radius))
        super().__init__(session, self.marker.structure)

class VerifyELCenterDialog(VerifyMarkerCenterDialog):
    def __init__(self, session, structure, maps, resolution, prefitted, initial_center, opaque_maps,
            show_sharpened_map, apply_symmetry):
        self._search_radius = None
        self.structure = structure
        self.maps = maps
        self.resolution = resolution
        self.show_sharpened_map = show_sharpened_map
        self.apply_symmetry = apply_symmetry
        self.prefitted = prefitted
        super().__init__(session, initial_center, opaque_maps)

    @property
    def check_models(self):
        return self.maps + [self.marker, self.structure]

    @property
    def instructions(self):
        return (
            "A transparent orange marker (model #%d) has been drawn to show the search volume and location. "
            "  Fits that place any part of the atomic structure within the marker sphere will be evaluated."
            "  The size of the search volume is based on the size of the structure and cannot be adjusted, "
            "but the search center can be moved by moving the marker, "
            "using any ChimeraX method for moving markers or models "
            '(e.g. the "move markers" right mouse mode in the Markers section of the toolbar).'
            '  When the position is satisfactory, click "%s."'
             % (self.marker_set_id, self.search_button_label)
        )

    def launch(self):
        if self.opaque_maps:
            for m, alpha in self.opaque_data.items():
                if not m.deleted:
                    rgba = list(m.rgba)
                    rgba[-1] = alpha
                    m.rgba = tuple(rgba)
        center = self.marker.scene_coord
        _run_emplace_local_command(self.session, self.structure, self.maps, self.resolution, self.prefitted,
            center, self.show_sharpened_map, self.apply_symmetry)

    @property
    def search_button_label(self):
        return "Start search"

    @property
    def search_radius(self):
        if self._search_radius is None:
            import numpy
            crds = self.structure.atoms.coords
            crd_min = numpy.amin(crds, axis=0)
            crd_max = numpy.amax(crds, axis=0)
            mid = (crd_min + crd_max) / 2
            self._search_radius = max(numpy.linalg.norm(crds-mid, axis=1))
        return self._search_radius

class VerifyStructureCenterDialog(VerifyCenterDialog):
    def __init__(self, session, initial_center, structure):
        self.structure = structure
        crd_mean = structure.atoms.coords.mean(0)
        import numpy
        structure.atoms.coords += numpy.array(initial_center) - structure.atoms.coords.mean(0)
        session.models.add([structure])
        super().__init__(session, structure)

class VerifyLFCenterDialog(VerifyStructureCenterDialog):
    search_button_text = "Start ligand fitting"

    def __init__(self, session, initial_center, ligand_fmt, ligand_value, receptor, map, chain_id, res_num,
            resolution, extent_type, extent_value, hbonds, clashes):
        self.session = session
        self.ligand_fmt = ligand_fmt
        self.ligand_value = ligand_value
        self.receptor = receptor
        self.map = map
        self.chain_id = chain_id
        self.res_num = res_num
        self.resolution = resolution
        self.extent_type = extent_type
        self.extent_value = extent_value
        self.hbonds = hbonds
        self.clashes = clashes
        from chimerax.core.models import Model
        if isinstance(self.ligand_value, Model):
            ligand = self.ligand_value.copy()
        else:
            from .ligand_fit import ligand_from_string
            ligand = ligand_from_string(session,
                LaunchLigandFitTool.ligand_fmt_to_prefix[ligand_fmt] + ligand_value)
        if extent_type == LaunchLigandFitTool.EXTENT_ANGSTROMS:
            extent_angstroms = extent_value
        else:
            from chimerax.geometry import distance
            longest = None
            for i, a1 in enumerate(ligand.atoms):
                for a2 in ligand.atoms[i+1:]:
                    d = distance(a1.coord, a2.coord)
                    if longest is None or d > longest:
                        longest = d
            if longest is None:
                longest = ligand_models[0].atoms[0].radius
            extent_angstroms = extent_value * longest
        from .ligand_fit import ijk_min_max
        ijk_min, ijk_max = ijk_min_max(map, initial_center, extent_angstroms)
        map.new_region(ijk_min, ijk_max, map.region[-1], adjust_step=False, adjust_voxel_limit=False)
        map.rendering_options.show_outline_box = True
        map.add_volume_change_callback(self._vol_change_cb)
        self.center = initial_center

        self.prev_mouse_mode = None
        for binding in session.ui.mouse_modes.bindings:
            if binding.matches('right', []):
                self.prev_mouse_mode = binding.mode
                break

        self.bounds_text = "Adjust search bounds"
        self.move_text = "Move example ligand"
        self.translate_text = "Translate scene"
        super().__init__(session, initial_center, ligand)
        from chimerax.core.commands import run
        run(session,
            f"view {ligand.atomspec} @<{extent_angstroms}; ui mousemode right 'translate selected models'; select {ligand.atomspec}")

    def add_custom_widgets(self, layout):
        super().add_custom_widgets(layout)
        from Qt.QtWidgets import QHBoxLayout, QButtonGroup, QGroupBox, QRadioButton
        button_area = QGroupBox("Right Mouse Function")
        button_area.setAlignment(Qt.AlignHCenter)
        layout.addWidget(button_area)
        b_layout = QHBoxLayout()
        button_area.setLayout(b_layout)
        from chimerax.core.commands import run
        self.bounds_button = QRadioButton(self.bounds_text)
        self.bounds_button.toggled.connect(lambda *args, run=run, ses=self.session:
            run(ses, "ui mousemode right 'crop volume'"))
        b_layout.addWidget(self.bounds_button)
        self.center_button = QRadioButton(self.move_text)
        self.center_button.toggled.connect(lambda *args, run=run, ses=self.session:
            run(ses, "ui mousemode right 'translate selected models'"))
        b_layout.addWidget(self.center_button)
        self.translate_button = QRadioButton(self.translate_text)
        self.translate_button.toggled.connect(lambda *args, run=run, ses=self.session:
            run(ses, "ui mousemode right translate"))
        b_layout.addWidget(self.translate_button)
        self.other_button = QRadioButton("(Other)")
        b_layout.addWidget(self.other_button)
        self.other_button.setEnabled(False)
        self.mouse_handler = self.session.triggers.add_handler("set mouse mode", self._mouse_mode_changed)

    @property
    def check_models(self):
        return [self.map, self.receptor, self.structure]

    def closeEvent(self, event):
        self.mouse_handler.remove()
        self.map.remove_volume_change_callback(self._vol_change_cb)
        return super().closeEvent(event)

    @property
    def instructions(self):
        return (
            "While the '%s' mouse mode (below) is active, you can move the ligand with the right mouse"
            " to place its center where you want the search focused.  The ligand must be selected"
            " (green outline) to be moved.  Once satified with the search focus, switch to the '%s'"
            " mouse mode to use the right mouse to adjust the bounds of the search area.  You can switch"
            " between centering/focusing and bounds adjustment as needed.  When satisified with the search"
            " area, click the '%s' button to fit the ligand." % (self.move_text, self.bounds_text,
                self.search_button_label)
        )

    def launch(self):
        # restore previous mouse mode
        if self.prev_mouse_mode is not None:
            from chimerax.core.commands import run
            run(self.session, f"ui mousemode right '{self.prev_mouse_mode.name}'")
        _run_ligand_fit_command(self.session, self.search_center, self.ligand_fmt, self.ligand_value,
            self.receptor, self.map, self.chain_id, self.res_num, self.resolution, None, None, self.hbonds,
            self.clashes)

    @property
    def search_button_label(self):
        return self.search_button_text

    @property
    def search_center(self):
        return self.structure.atoms.scene_coords.mean(0)

    def _mouse_mode_changed(self, trig_name, trig_data):
        button, modifiers, mode = trig_data
        all_buttons = [self.center_button, self.bounds_button, self.translate_button, self.other_button]
        for b in all_buttons:
            b.blockSignals(True)
        try:
            if mode.name == 'translate selected models':
                self.center_button.setChecked(True)
                self.other_button.setHidden(True)
            elif mode.name == 'crop volume':
                self.bounds_button.setChecked(True)
                self.other_button.setHidden(True)
            elif mode.name == 'translate':
                self.translate_button.setChecked(True)
                self.other_button.setHidden(True)
            else:
                self.other_button.setText(f"({mode.name})")
                self.other_button.setHidden(False)
                self.other_button.setChecked(True)
        finally:
            for b in all_buttons:
                b.blockSignals(False)

    def _vol_change_cb(self, vol, reason):
        if reason != "region changed" or vol != self.map:
            return
        # Ensure region still encloses search center
        vxyz = vol.scene_position.inverse() * self.center
        center_ijk = vol.data.xyz_to_ijk(vxyz)
        ijk_min, ijk_max, ijk_step = vol.region
        # avoid directly modifying
        import sys
        ijk_min = list(ijk_min[:])
        ijk_max = list(ijk_max[:])
        violated = False
        for index in range(3):
            c_val = center_ijk[index]
            if c_val < ijk_min[index]:
                violated = True
                while ijk_min[index] > c_val:
                    ijk_min[index] -= 1
            elif c_val > ijk_max[index]:
                violated = True
                while ijk_max[index] < c_val:
                    ijk_max[index] += 1
        if violated:
            vol.new_region(ijk_min, ijk_max, ijk_step, adjust_step=False, adjust_voxel_limit=False)
            self.session.logger.status("Search center must be within volume box; clamping box",
                color="medium purple", blank_after=5)
        else:
            self.session.logger.status("")

class PickBlobDialog(QDialog):
    instructions = "Right click on the volume \"blob\" where you want the ligand placed. " \
        " A yellow marker will appear indicating where the search will be focused. " \
        " Clicking again will replace the marker if desired."

    def __init__(self, session, verify_center, *non_center_args):
        super().__init__()
        self.session = session
        self.verify_center = verify_center
        self.non_center_args = non_center_args
        ligand_fmt, ligand_value, receptor, map, chain_id, res_num, resolution, extent_type, \
            extent_value, hbonds, clashes = non_center_args

        from Qt.QtWidgets import QVBoxLayout, QLabel
        layout = QVBoxLayout()
        self.setLayout(layout)
        instructions = QLabel(self.instructions)
        instructions.setWordWrap(True)
        instructions.setAlignment(Qt.AlignCenter)
        layout.addWidget(instructions)

        self.prev_mouse_mode = None
        for binding in session.ui.mouse_modes.bindings:
            if binding.matches('right', []):
                self.prev_mouse_mode = binding.mode
                break

        self.pick_text = "Pick volume blob"
        self.translate_text = "Translate scene"

        from Qt.QtWidgets import (
            QHBoxLayout, QButtonGroup, QGroupBox, QRadioButton, QDoubleSpinBox, QCheckBox
        )
        button_area = QGroupBox("Right Mouse Function")
        button_area.setAlignment(Qt.AlignHCenter)
        layout.addWidget(button_area)
        b_layout = QHBoxLayout()
        button_area.setLayout(b_layout)
        from chimerax.core.commands import run
        self.blob_button = QRadioButton(self.pick_text)
        self.blob_button.toggled.connect(lambda *args, run=run, ses=self.session:
            run(ses, "ui mousemode right 'mark maximum'"))
        b_layout.addWidget(self.blob_button)
        self.translate_button = QRadioButton(self.translate_text)
        self.translate_button.toggled.connect(lambda *args, run=run, ses=self.session:
            run(ses, "ui mousemode right translate"))
        b_layout.addWidget(self.translate_button)
        self.other_button = QRadioButton("(Other)")
        b_layout.addWidget(self.other_button)
        self.other_button.setEnabled(False)
        self.mouse_handler = self.session.triggers.add_handler("set mouse mode", self._mouse_mode_changed)

        hide_density_layout = QHBoxLayout()
        hide_density_layout.setSpacing(2)
        layout.addLayout(hide_density_layout)
        hide_density_layout.addStretch(1)
        self.hide_density = QCheckBox("Hide density within ")
        self.hide_density.setChecked(False)
        self.hide_density.toggled.connect(self._hide_density)
        hide_density_layout.addWidget(self.hide_density)
        self.hide_dist = QDoubleSpinBox()
        self.hide_dist.setRange(0.5, 5.0)
        self.hide_dist.setDecimals(1)
        self.hide_dist.setSingleStep(0.1)
        self.hide_dist.setValue(1.8)
        hide_density_layout.addWidget(self.hide_dist)
        hide_density_layout.addWidget(QLabel("\N{ANGSTROM SIGN} of existing structure"))
        hide_density_layout.addStretch(1)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        bbox = qbbox(qbbox.Cancel)
        bbox.addButton("Adjust search zone" if verify_center else VerifyLFCenterDialog.search_button_text,
            bbox.AcceptRole)
        bbox.accepted.connect(self.launch)
        bbox.rejected.connect(self.close)
        layout.addWidget(bbox)

        self.receptor, self.map = self.check_models = [receptor, map]
        from chimerax.core.models import REMOVE_MODELS
        self.remove_models_handler = session.triggers.add_handler(REMOVE_MODELS, self._check_still_valid)
        from chimerax.atomic import get_triggers
        self.new_marker_handler = get_triggers().add_handler('changes', self._new_marker_check)
        self._current_marker = None
        self._creating_markers = False

        self.show()

        from chimerax.core.commands import run
        run(session, f"ui mousemode right 'mark maximum'")

    def closeEvent(self, event):
        self.mouse_handler.remove()
        self.remove_models_handler.remove()
        self.new_marker_handler.remove()
        return super().closeEvent(event)

    def launch(self):
        self.hide()
        self.session.ui.processEvents()
        # restore previous mouse mode
        if self.prev_mouse_mode is not None:
            from chimerax.core.commands import run
            run(self.session, f"ui mousemode right '{self.prev_mouse_mode.name}'")
        if self._current_marker and not self._current_marker.deleted:
            center = self._current_marker.scene_coord
            self._current_marker.structure.delete_atom(self._current_marker)
        else:
            self.show()
            from chimerax.ui import tool_user_error
            return tool_user_error("No volume blob picked")
        if self.verify_center:
            VerifyLFCenterDialog(self.session, center, *self.non_center_args)
        else:
            _run_ligand_fit_command(self.session, center, *self.non_center_args)
        self.close()

    def _check_still_valid(self, trig_name, removed_models):
        for rm in removed_models:
            if rm in self.check_models:
                self.close()
                break

    def _hide_density(self, hide):
        from chimerax.core.commands import run
        if hide:
            run(self.session, f"volume zone {self.map.atomspec} near #!{self.receptor.id_string} range {self.hide_dist.value()} invert true")
        else:
            run(self.session, f"volume unzone {self.map.atomspec}")

    def _mouse_mode_changed(self, trig_name, trig_data):
        button, modifiers, mode = trig_data
        all_buttons = [self.blob_button, self.translate_button, self.other_button]
        for b in all_buttons:
            b.blockSignals(True)
        try:
            if mode.name == 'mark maximum':
                self.blob_button.setChecked(True)
                self.other_button.setHidden(True)
                self._creating_markers = True
            elif mode.name == 'translate':
                self.translate_button.setChecked(True)
                self.other_button.setHidden(True)
                self._creating_markers = False
            else:
                self.other_button.setText(f"({mode.name})")
                self.other_button.setHidden(False)
                self.other_button.setChecked(True)
                self._creating_markers = False
        finally:
            for b in all_buttons:
                b.blockSignals(False)

    def _new_marker_check(self, trig_name, trig_data):
        if not self._creating_markers:
            return
        from chimerax.markers import MarkerSet
        for a in trig_data.created_atoms():
            if isinstance(a.structure, MarkerSet):
                if self._current_marker and not self._current_marker.deleted:
                    self._current_marker.structure.delete_atom(self._current_marker)
                self._current_marker = a
                break

class LaunchLigandFitTool(ToolInstance):
    #help = "help:user/tools/localemfitting.html"
    help = None

    CENTER_BLOB = "picked volume blob"
    CENTER_MODEL = "center of model..."
    CENTER_SELECTION = "center of selection"
    CENTER_VIEW = "center of view"
    CENTER_XYZ = "specified xyz position..."
    CENTERING_METHODS = [CENTER_BLOB, CENTER_MODEL, CENTER_SELECTION, CENTER_VIEW, CENTER_XYZ]

    EXTENT_LENGTH = "ligand lengths"
    EXTENT_ANGSTROMS = "angstroms"
    EXTENT_METHODS = [EXTENT_LENGTH, EXTENT_ANGSTROMS]

    LIGAND_FMT_CCD = "CCD identifier"
    LIGAND_FMT_MODEL = "existing structure"
    LIGAND_FMT_PUBCHEM = "PubChem identifier"
    LIGAND_FMT_SMILES = "SMILES string"
    LIGAND_FORMATS = [LIGAND_FMT_CCD, LIGAND_FMT_MODEL, LIGAND_FMT_PUBCHEM, LIGAND_FMT_SMILES]
    ligand_fmt_to_prefix = {LIGAND_FMT_CCD: "ccd:", LIGAND_FMT_MODEL: "", LIGAND_FMT_PUBCHEM: "pubchem:",
        LIGAND_FMT_SMILES: "smiles:"}

    def __init__(self, session, tool_name):
        super().__init__(session, tool_name)
        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self, close_destroys=False)
        parent = tw.ui_area

        if not hasattr(self.__class__, 'settings'):
            self.__class__.settings = LaunchLigandFitSettings(session, "launch ligandFit")

        from Qt.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QWidget, QPushButton, QMenu, QLineEdit
        from Qt.QtWidgets import QCheckBox, QGridLayout, QGroupBox, QStackedWidget
        from Qt.QtGui import QDoubleValidator, QIntValidator
        from Qt.QtCore import Qt
        layout = QVBoxLayout()
        parent.setLayout(layout)
        layout.setSpacing(1)

        ligand_layout = QHBoxLayout()
        ligand_layout.setContentsMargins(2,2,2,2)
        layout.addLayout(ligand_layout)
        ligand_layout.addWidget(QLabel("Ligand from:"))
        self.ligand_fmt_to_index = {}
        self.ligand_fmt_button = QPushButton()
        ligand_layout.addWidget(self.ligand_fmt_button)
        fmt_menu = QMenu(self.ligand_fmt_button)
        fmt_menu.triggered.connect(self._fmt_menu_cb)
        self.ligand_fmt_button.setMenu(fmt_menu)
        self.ligand_stack = QStackedWidget()
        ligand_layout.addWidget(self.ligand_stack, stretch=1)
        from chimerax.atomic.widgets import AtomicStructureMenuButton
        for ligand_fmt in self.LIGAND_FORMATS:
            fmt_menu.addAction(ligand_fmt)
            if ligand_fmt == self.LIGAND_FMT_MODEL:
                widget = AtomicStructureMenuButton(session)
            else:
                widget = QLineEdit()
                widget.setPlaceholderText("Enter " + ligand_fmt
                    + (" (a.k.a. PDB 3- or 5-letter residue code)"
                    if ligand_fmt == self.LIGAND_FMT_CCD else ""))
            self.ligand_fmt_to_index[ligand_fmt] = self.ligand_stack.addWidget(widget)
        self._update_fmt_widgets(self.settings.ligand_format)

        spec_centering_widget = QWidget()
        layout.addWidget(spec_centering_widget, alignment=Qt.AlignCenter)
        spec_layout = QHBoxLayout()
        spec_layout.setContentsMargins(2,2,2,2)
        spec_centering_widget.setLayout(spec_layout)
        spec_layout.addWidget(QLabel("With chain ID"))
        self.chain_id_entry = QLineEdit()
        self.chain_id_entry.setMaximumWidth(5 * self.chain_id_entry.fontMetrics().averageCharWidth())
        self.chain_id_entry.editingFinished.connect(self._update_res_num_widget)
        spec_layout.addWidget(self.chain_id_entry)
        spec_layout.addWidget(QLabel("and residue number"))
        self.res_num_entry = QLineEdit()
        self.res_num_entry.setMaximumWidth(5 * self.res_num_entry.fontMetrics().averageCharWidth())
        self.res_num_entry.setValidator(QIntValidator())
        spec_layout.addWidget(self.res_num_entry)

        non_lig_layout = QHBoxLayout()
        layout.addLayout(non_lig_layout)
        receptor_centering_widget = QWidget()
        non_lig_layout.addWidget(receptor_centering_widget)
        receptor_layout = QHBoxLayout()
        receptor_layout.setSpacing(3)
        receptor_centering_widget.setLayout(receptor_layout)
        receptor_layout.addWidget(QLabel("Receptor"))
        self.receptor_menu = AtomicStructureMenuButton(session)
        self.receptor_menu.value_changed.connect(self._update_spec_widgets)
        receptor_layout.addWidget(self.receptor_menu)
        map_info_centering_widget = QWidget()
        non_lig_layout.addWidget(map_info_centering_widget)
        map_info_layout = QVBoxLayout()
        map_info_layout.setSpacing(0)
        map_info_centering_widget.setLayout(map_info_layout)
        map_centering_widget = QWidget()
        map_info_layout.addWidget(map_centering_widget, alignment=Qt.AlignBottom|Qt.AlignHCenter)
        map_layout = QHBoxLayout()
        map_layout.setContentsMargins(0,0,0,0)
        map_layout.setSpacing(3)
        map_centering_widget.setLayout(map_layout)
        map_layout.addWidget(QLabel("Map:"))
        from chimerax.map import Volume
        from chimerax.ui.widgets import ModelMenuButton
        self.map_menu = ModelMenuButton(session, class_filter=Volume)
        map_layout.addWidget(self.map_menu)
        res_centering_widget = QWidget()
        map_info_layout.addWidget(res_centering_widget, alignment=Qt.AlignTop|Qt.AlignHCenter)
        res_layout = QHBoxLayout()
        res_layout.setContentsMargins(0,0,0,0)
        res_centering_widget.setLayout(res_layout)
        res_layout.addWidget(QLabel("Resolution:"))
        self.resolution_entry = QLineEdit()
        self.resolution_entry.setMaximumWidth(5 * self.resolution_entry.fontMetrics().averageCharWidth())
        self.resolution_entry.setValidator(QDoubleValidator(0.001, 1000.0, 6))
        res_layout.addWidget(self.resolution_entry)
        self._update_spec_widgets()

        layout.addStretch(1)

        centering_layout = QHBoxLayout()
        centering_layout.setSpacing(1)
        centering_layout.addStretch(1)
        layout.addLayout(centering_layout)
        centering_tip = '''How to specify the center of the fitting search.  Choices are:

%s - A blob of density chosen interactively (after clicking OK or Apply on this dialog).

%s — If the center of rotation is being displayed ("cofr showPivot true") use that.  Otherwise,
    if the center of rotation is a fixed point ("cofr fixed") use that.  If neither of those is
    true, use the midpoint of where the center of the window intersects the front and back of
    the bounding box of the map.

%s — The center of a particular model, frequently the map, or the structure to be fitted once
    it has been approximately positioned.

%s - The center of the bounding box enclosing currently selected objects.

%s — A specific X/Y/Z position, given in angstroms relative to the origin of the map.
        ''' % (self.CENTER_BLOB, self.CENTER_VIEW.rstrip('.'), self.CENTER_MODEL.rstrip('.'),
            self.CENTER_SELECTION.rstrip('.'), self.CENTER_XYZ.rstrip('.'))
        centering_label = QLabel("Center search at")
        centering_label.setToolTip(centering_tip)
        centering_layout.addWidget(centering_label, alignment=Qt.AlignRight)
        self.centering_button = QPushButton()
        self.centering_button.setToolTip(centering_tip)
        centering_layout.addWidget(self.centering_button)
        centering_menu = QMenu(self.centering_button)
        for method in self.CENTERING_METHODS:
            centering_menu.addAction(method)
        centering_menu.triggered.connect(lambda act: self._set_centering_method(act.text()))
        self.centering_button.setMenu(centering_menu)
        self.xyz_area = QWidget()
        xyz_layout = QHBoxLayout()
        xyz_layout.setSpacing(1)
        self.xyz_area.setLayout(xyz_layout)
        self.xyz_widgets = []
        for lab in ["X", " Y", " Z"]:
            xyz_layout.addWidget(QLabel(lab), alignment=Qt.AlignRight)
            entry = QLineEdit()
            entry.setValidator(QDoubleValidator())
            entry.setAlignment(Qt.AlignCenter)
            entry.setMaximumWidth(50)
            entry.setText("0")
            xyz_layout.addWidget(entry, alignment=Qt.AlignLeft)
            self.xyz_widgets.append(entry)
        self.model_menu = ModelMenuButton(session)
        self.blob_label = QLabel("(chosen after OK/Apply)")
        centering_layout.addWidget(self.model_menu)
        centering_layout.addWidget(self.xyz_area)
        centering_layout.addWidget(self.blob_label)
        centering_layout.addStretch(1)

        extent_layout = QHBoxLayout()
        extent_layout.setSpacing(1)
        layout.addLayout(extent_layout)
        extent_tip = '''How to specify the extent of the search from the center (along cell axes).
Choices are:

%s - The longest atom-to-atom length in the ligand

%s — A fixed amount in angstroms
''' % (self.EXTENT_LENGTH, self.EXTENT_ANGSTROMS)
        extent_layout.addStretch(1)
        extent_layout.addWidget(QLabel("Search region within "))
        self.extent_entry = entry = QLineEdit()
        entry.setValidator(QDoubleValidator())
        entry.setAlignment(Qt.AlignRight)
        entry.setMaximumWidth(30)
        entry.setText("%g" % self.settings.extent_value)
        extent_layout.addWidget(entry)
        self.extent_button = QPushButton()
        self.extent_button.setToolTip(extent_tip)
        extent_layout.addWidget(self.extent_button)
        extent_menu = QMenu(self.extent_button)
        for method in self.EXTENT_METHODS:
            extent_menu.addAction(method)
        extent_menu.triggered.connect(lambda act: self._set_extent_method(act.text()))
        self.extent_button.setMenu(extent_menu)
        extent_layout.addWidget(QLabel(" of center along cell axes"))
        extent_layout.addStretch(1)

        self._set_centering_method()
        self._extent_values = [None] * len(self.EXTENT_METHODS)
        self._set_extent_method()

        layout.addStretch(1)

        checkbox_area = QWidget()
        layout.addWidget(checkbox_area, alignment=Qt.AlignCenter)
        checkbox_layout = QVBoxLayout()
        checkbox_layout.setContentsMargins(0,0,0,0)
        checkbox_area.setLayout(checkbox_layout)
        self.verify_center_checkbox = QCheckBox("Interactively adjust search region before fitting")
        self.verify_center_checkbox.setChecked(True)
        checkbox_layout.addWidget(self.verify_center_checkbox, alignment=Qt.AlignLeft)
        self.show_hbonds_checkbox = QCheckBox("Show H-bonds formed by fit ligand")
        self.show_hbonds_checkbox.setChecked(True)
        checkbox_layout.addWidget(self.show_hbonds_checkbox, alignment=Qt.AlignLeft)
        self.show_clashes_checkbox = QCheckBox("Show clashes with fit ligand")
        self.show_clashes_checkbox.setChecked(True)
        checkbox_layout.addWidget(self.show_clashes_checkbox, alignment=Qt.AlignLeft)

        layout.addSpacing(10)

        layout.addWidget(PhenixCitation(session, tool_name, "ligandfit"), alignment=Qt.AlignCenter)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        self.bbox = bbox = qbbox(qbbox.Ok | qbbox.Apply | qbbox.Close | qbbox.Help)
        bbox.accepted.connect(self.launch_ligand_fit)
        bbox.button(qbbox.Apply).clicked.connect(lambda *args: self.launch_ligand_fit(apply=True))
        bbox.rejected.connect(self.delete)
        if self.help:
            from chimerax.core.commands import run
            bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help " + self.help))
        else:
            bbox.button(qbbox.Help).setEnabled(False)
        layout.addWidget(bbox)

        tw.manage(placement=None)

    def launch_ligand_fit(self, apply=False):
        ligand_fmt = self.ligand_fmt_button.text()
        ligand_widget = self.ligand_stack.currentWidget()
        if ligand_fmt == self.LIGAND_FMT_MODEL:
            ligand_value = ligand_widget.value
            if not ligand_value:
                raise UserError("No ligand model specified")
        else:
            ligand_value = ligand_widget.text().strip()
            if not ligand_value:
                raise UserError("No " + ligand_fmt + " text provided")

        receptor = self.receptor_menu.value
        if not receptor:
            raise UserError("Must specify a receptor structure")
        map = self.map_menu.value
        if map:
            if self.resolution_entry.hasAcceptableInput():
                resolution = float(self.resolution_entry.text())
            else:
                raise UserError("Must specify a resolution value for the map")
        else:
            raise UserError("Must specify map for fitting")
        chain_id = self.chain_id_entry.text().strip()
        if self.res_num_entry.text().strip():
            if self.res_num_entry.hasAcceptableInput():
                res_num = int(self.res_num_entry.text())
            else:
                raise UserError("Residue number must be an integer")
        else:
            res_num = None
        if not self.extent_entry.hasAcceptableInput():
            raise UserError("Search-extent value not a valid number")
        self.settings.extent_value = extent_value = float(self.extent_entry.text())
        self.settings.extent_type = extent_type = self.extent_button.text()

        if extent_value <= 0:
            raise UserError("Search-extent value must be a positive number")

        non_center_args = (ligand_fmt, ligand_value, receptor, map, chain_id, res_num,
            resolution, extent_type, extent_value, self.show_hbonds_checkbox.isChecked(),
            self.show_clashes_checkbox.isChecked())

        self.settings.search_center = method = self.centering_button.text()
        if not apply:
            self.display(False)
        verify_center = self.verify_center_checkbox.isChecked()

        if method == self.CENTER_BLOB:
            from chimerax.core.errors import LimitationError
            #raise LimitationError("Blob-picking centering not yet implemented")
            # Probably needs to subclass VerifyCenterDialog, so that (among other things)
            # triggers hold a reference to the dialog so that it isn't immediately destroyed
            return PickBlobDialog(self.session, verify_center, *non_center_args)
        elif method == self.CENTER_XYZ:
            center = [float(widget.text()) for widget in self.xyz_widgets]
        elif method == self.CENTER_MODEL:
            centering_model = self.model_menu.value
            if centering_model is None:
                raise UserError("No model chosen for specifying search center")
            bnds = centering_model.bounds()
            if bnds is None:
                raise UserError("No part of model for specifying search center is displayed")
            center = bnds.center()
        elif method == self.CENTER_VIEW:
            # If pivot point shown or using fixed center of rotation, use that.
            # Otherwise, midpoint where center of window intersects front and back of halfmap bounding box.
            view_center = None
            mv = self.session.main_view
            for d in mv.drawing.child_drawings():
                if d.__class__.__name__ == "PivotIndicator":
                    view_center = d.position.origin()
                    break
            else:
                if mv.center_of_rotation_method == "fixed":
                    view_center = mv.center_of_rotation
            if view_center is None:
                from chimerax.map import Volume
                shown_vols = [v for v in self.session.models if isinstance(v, Volume) and v.display]
                if len(shown_vols) == 1:
                    view_map = shown_vols[0]
                else:
                    view_map = maps[0]
                from .emplace_local import view_box, ViewBoxError
                try:
                    view_center = view_box(self.session, view_map)
                except ViewBoxError as e:
                    raise UserError(str(e))
            center = view_center
        elif method == self.CENTER_SELECTION:
            if self.session.selection.empty():
                raise UserError("Nothing selected")
            from chimerax.atomic import selected_atoms
            sel_atoms = selected_atoms(self.session)
            from chimerax.geometry import point_bounds, union_bounds
            atom_bbox = point_bounds(sel_atoms.scene_coords)
            atom_models = set(sel_atoms.unique_structures)
            bbox = union_bounds([atom_bbox]
                + [m.bounds() for m in self.session.selection.models() if m not in atom_models])
            if bbox is None:
                raise UserError("No bounding box for selected items")
            center = bbox.center()
        else:
            raise AssertionError("Unknown centering method")
        if verify_center:
            VerifyLFCenterDialog(self.session, center, *non_center_args)
        else:
            _run_ligand_fit_command(self.session, center, *non_center_args)

    def _fmt_menu_cb(self, action):
        self._update_fmt_widgets(action.text())

    def _set_centering_method(self, method=None):
        if method is None:
            method = self.settings.search_center
            if method not in self.CENTERING_METHODS:
                method = self.CENTER_BLOB
        self.centering_button.setText(method)
        self.xyz_area.setHidden(True)
        self.model_menu.setHidden(True)
        self.blob_label.setHidden(True)
        if method == self.CENTER_XYZ:
            self.xyz_area.setHidden(False)
        elif method == self.CENTER_MODEL:
            self.model_menu.setHidden(False)
        elif method == self.CENTER_BLOB:
            self.blob_label.setHidden(False)

    def _set_extent_method(self, method=None):
        if method is None:
            method = self.settings.extent_type
            if method not in self.EXTENT_METHODS:
                method = self.EXTENT_LENGTH
                new_value = 1.1
            else:
                new_value = self.settings.extent_value
        else:
            # we're switching the method; remember the old value if valid and restore from previous if known
            if self.extent_entry.hasAcceptableInput():
                self._extent_values[self.EXTENT_METHODS.index(self.extent_button.text())] = float(
                    self.extent_entry.text())
            prev_extent = self._extent_values[self.EXTENT_METHODS.index(method)]
            if prev_extent is None:
                new_value = 1.1 if method == self.EXTENT_LENGTH else 10
            else:
                new_value = prev_extent
        self.extent_button.setText(method)
        self.extent_entry.setText("%g" % new_value)

    def _update_fmt_widgets(self, fmt):
        self.ligand_fmt_button.setText(fmt)
        self.ligand_stack.setCurrentIndex(self.ligand_fmt_to_index[fmt])
        h = self.ligand_stack.currentWidget().sizeHint().height()
        self.ligand_stack.setFixedHeight(h)

    def _update_res_num_widget(self, receptor=None):
        if receptor is None:
            receptor = self.receptor_menu.value
            if not receptor:
                return
        chain_id = self.chain_id_entry.text()
        residues = receptor.residues
        chain_residues = residues.filter(residues.chain_ids == chain_id)
        if chain_residues:
            next_res_num = max(chain_residues.numbers) + 1
        else:
            next_res_num = 1
        self.res_num_entry.setText(str(next_res_num))

    def _update_spec_widgets(self):
        receptor = self.receptor_menu.value
        if not receptor:
            return
        if self.res_num_entry.text() or self.chain_id_entry.text():
            return
        if receptor.num_chains != 1:
            return
        chain = receptor.chains[0]
        self.chain_id_entry.setText(chain.chain_id)
        self._update_res_num_widget(receptor)

class LaunchLigandFitSettings(Settings):
    AUTO_SAVE = {
        'ligand_format': LaunchLigandFitTool.LIGAND_FMT_CCD,
        'search_center': LaunchLigandFitTool.CENTER_BLOB,
        'extent_type': LaunchLigandFitTool.EXTENT_LENGTH,
        'extent_value': 1.1,
    }

def _run_emplace_local_command(session, structure, maps, resolution, prefitted, center, show_sharpened_map,
        apply_symmetry):
    from chimerax.core.commands import run, concise_model_spec, BoolArg, StringArg
    from chimerax.map import Volume
    cmd = "phenix emplaceLocal %s mapData %s resolution %g center %g,%g,%g showSharpenedMap %s" \
        " applySymmetry %s" % (
        structure.atomspec, concise_model_spec(session, maps, relevant_types=Volume, allow_empty_spec=False),
        resolution, *center, BoolArg.unparse(show_sharpened_map), BoolArg.unparse(apply_symmetry))
    if prefitted:
        from chimerax.atomic import AtomicStructure
        cmd += " prefitted %s" % concise_model_spec(session, prefitted, relevant_types=AtomicStructure,
            allow_empty_spec=False)
    run(session, cmd)

def _run_ligand_fit_command(session, center, ligand_fmt, ligand_value, receptor, map, chain_id, res_num,
        resolution, extent_type, extent_value, hbonds, clashes):
    from chimerax.core.commands import run, StringArg, BoolArg
    from chimerax.map import Volume
    LLFT = LaunchLigandFitTool
    lig_arg = "%s%s" % (LaunchLigandFitTool.ligand_fmt_to_prefix[ligand_fmt],
        (ligand_value.atomspec if ligand_fmt == LLFT.LIGAND_FMT_MODEL else ligand_value))
    if extent_type is None:
        extent_arg = ""
    else:
        extent_arg =  " extentType %s extentValue %g" % (
            ("length" if extent_type == LaunchLigandFitTool.EXTENT_LENGTH else "angstroms"), extent_value)
    cmd = "phenix ligandFit %s ligand %s center %g,%g,%g inMap %s resolution %g%s " \
        " hbonds %s clashes %s" % (receptor.atomspec, StringArg.unparse(lig_arg), *center,
        map.atomspec, resolution, extent_arg, BoolArg.unparse(hbonds), BoolArg.unparse(clashes))
    if chain_id:
        cmd += " chain " + chain_id
    if res_num is not None:
        cmd += " residueNum " + str(res_num)
    run(session, cmd)

class FitLoopsResultsSettings(Settings):
    AUTO_SAVE = {
        'last_advised': None
    }

class FitLoopsResultsViewer(ToolInstance):

    #help = "help:user/tools/waterplacement.html#waterlist"

    def __init__(self, session, model=None, fit_info=None, map=None):
        # if 'model' is None, we are being restored from a session and _finalize_init() will be called later
        super().__init__(session, "Fit Loops Results")
        if model is None:
            return
        self._finalize_init(model, fit_info, map)

    def _finalize_init(self, model, fit_info, map, *, session_info=None):
        self.model = model
        self.fit_info = fit_info
        self.map = map

        from chimerax.core.models import REMOVE_MODELS
        self.handlers = [
            self.session.triggers.add_handler(REMOVE_MODELS, self._models_removed_cb),
        ]

        # change any sphere representations into stick
        if not session_info:
            check_atoms = self.model.atoms
            check_spheres = check_atoms.filter(check_atoms.draw_modes == check_atoms.SPHERE_STYLE)
            check_spheres.draw_modes = check_atoms.STICK_STYLE

        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self, close_destroys=False)
        parent = tw.ui_area

        from Qt.QtWidgets import QHBoxLayout, QButtonGroup, QVBoxLayout, QRadioButton, QCheckBox
        from Qt.QtWidgets import QPushButton, QLabel, QToolButton, QGridLayout
        layout = QVBoxLayout()
        layout.setContentsMargins(2,2,2,2)
        layout.setSpacing(0)
        parent.setLayout(layout)

        self.table = self._build_table()
        layout.addWidget(self.table, stretch=1)

        self.tool_window.manage('side')

    def delete(self):
        for handler in self.handlers:
            handler.remove()
        self.model = self.map = None
        super().delete()

    def _build_table(self):
        class FLInfo:
            def __init__(self, info, model):
                for k, v in info.items():
                    setattr(self, k, v)
                if self.successful:
                    self.successful = "\N{CHECK MARK}"
                    res_offset = 0
                else:
                    self.cc = None
                    self.successful = ""
                    res_offset = 1
                residues = []
                for chain in model.chains:
                    if chain.chain_id == self.chain_id:
                        for r in chain.existing_residues:
                            if r.number >= self.start_residue - res_offset \
                            and r.number <= self.end_residue + res_offset:
                                residues.append(r)
                from chimerax.atomic import Residues
                self.residues = Residues(residues)
        from chimerax.ui.widgets import ItemTable
        table = ItemTable()
        chain_col = table.add_column("Chain", "chain_id")
        table.add_column("Start", "start_residue")
        table.add_column("End", "end_residue")
        table.add_column("Success", "successful")
        table.add_column("CC", "cc", format="%.3f", balloon="Correlation coefficient with map")
        table.add_column("Sequence", "gap_sequence")
        table.data = [FLInfo(info, self.model)
            for info in self.fit_info if info['segment_number'] is not None]
        table.launch(select_mode=table.SelectionMode.SingleSelection)
        table.sort_by(chain_col, table.SORT_ASCENDING)
        table.selection_changed.connect(self._new_selection)
        if len(table.data) == 1:
            table.selected = table.data
        else:
            if not hasattr(self.__class__, 'settings'):
                self.__class__.settings = FitLoopsResultsSettings(self.session, "fit loops results")
            last = self.settings.last_advised
            from time import time
            now = self.settings.last_advised = time()
            if last is None or now - last >= 777700: # about 3 months
                from Qt.QtWidgets import QMessageBox
                msg = QMessageBox()
                msg.setWindowTitle("Table Selection")
                msg.setText("Select a row in the results table to focus in on that part of the structure")
                msg.exec()
        return table

    def _models_removed_cb(self, trig_name, trig_data):
        if self.model in trig_data:
            self.delete()

    def _new_selection(self, selected, unselected):
        if selected:
            residues = selected[0].residues
            if residues:
                from chimerax.core.commands import run
                from chimerax.atomic import concise_residue_spec
                # display things if necessary
                with self.session.undo.aggregate("Fit Loops results table row"):
                    if not residues.ribbon_displays.any() and not residues.atoms.displays.any():
                        if self.model.residues.ribbon_displays.any():
                            run(self.session, "cartoon %s" % spec)
                        else:
                            run(self.session, "display %s" % spec)
                    spec = concise_residue_spec(self.session, residues)
                    run(self.session, "sel %s; view sel" % spec)
            else:
                self.session.logger.status("No residues to view for this row")
