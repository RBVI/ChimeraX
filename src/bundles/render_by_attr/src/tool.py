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
from chimerax.atomic import Atom

from Qt.QtCore import Qt

class RenderByAttrTool(ToolInstance):

    help = "help:user/tools/render.html"

    NO_ATTR_TEXT = "choose attr"

    RENDER_COLORS = "Colors"
    RENDER_RADII = "Radii"
    RENDER_WORMS = "Worms"

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)
        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self, statusbar=True)
        parent = tw.ui_area
        from Qt.QtWidgets import QVBoxLayout, QHBoxLayout, QDialogButtonBox, QPushButton, QMenu, QLabel
        from Qt.QtWidgets import QTabWidget, QWidget, QCheckBox, QLineEdit
        from Qt.QtGui import QDoubleValidator
        from Qt.QtCore import Qt
        overall_layout = QVBoxLayout()
        overall_layout.setContentsMargins(3,0,3,0)
        overall_layout.setSpacing(12)
        parent.setLayout(overall_layout)

        target_layout = QHBoxLayout()
        target_layout.setSpacing(3)
        overall_layout.addLayout(target_layout)

        attribute_layout = QVBoxLayout()
        attribute_layout.setSpacing(0)
        target_menu_widget = QWidget()
        target_menu_layout = QHBoxLayout()
        target_menu_layout.setSpacing(2)
        target_menu_widget.setLayout(target_menu_layout)
        attribute_layout.addWidget(target_menu_widget, alignment=Qt.AlignBottom, stretch=1)
        target_menu_layout.addWidget(QLabel("Attributes of"), alignment=Qt.AlignRight)
        self.target_menu_button = QPushButton()
        menu = QMenu()
        menu.triggered.connect(self._new_target)
        self.target_menu_button.setMenu(menu)
        target_menu_layout.addWidget(self.target_menu_button, alignment=Qt.AlignLeft)
        attr_menu_widget = QWidget()
        attr_menu_layout = QHBoxLayout()
        attr_menu_layout.setSpacing(2)
        attr_menu_widget.setLayout(attr_menu_layout)
        attribute_layout.addWidget(attr_menu_widget, alignment=Qt.AlignTop, stretch=1)
        attr_menu_layout.addWidget(QLabel("Attribute:"), alignment=Qt.AlignRight)
        self.attr_menu_button = QPushButton()
        menu = QMenu()
        menu.triggered.connect(self._new_attr)
        menu.aboutToShow.connect(self._update_attr_menu)
        self.attr_menu_button.setMenu(menu)
        attr_menu_layout.addWidget(self.attr_menu_button, alignment=Qt.AlignLeft)
        model_list_layout = QVBoxLayout()
        model_list_layout.addWidget(QLabel("Models"), alignment=Qt.AlignBottom)
        from chimerax.ui.widgets import ModelListWidget, MarkedHistogram, PaletteChooser
        class SmallerModelListWidget(ModelListWidget):
            def sizeHint(self):
                sh = super().sizeHint()
                sh.setHeight(sh.height() // 2)
                sh.setWidth(sh.width() * 2 // 3)
                return sh
        self.model_list = SmallerModelListWidget(session, filter_func=self._filter_model)
        self.model_list.value_changed.connect(self._models_changed)
        model_list_layout.addWidget(self.model_list, alignment=Qt.AlignTop)
        target_layout.addLayout(attribute_layout)
        target_layout.addLayout(model_list_layout)

        self.mode_widget = QTabWidget()
        overall_layout.addWidget(self.mode_widget)

        render_tab = QWidget()
        render_tab_layout = QVBoxLayout()
        render_tab.setLayout(render_tab_layout)
        render_tab_layout.setSpacing(1)
        self.render_histogram = rh = MarkedHistogram(min_label=True, max_label=True, status_line=tw.status,
            select_callback=self._render_sel_marker_cb)
        render_tab_layout.addWidget(rh)
        self.render_markers = {}
        self.render_type_widgets = {}
        self.render_type_widget = QTabWidget()
        self.render_type_widget.setTabBarAutoHide(True)
        render_tab_layout.addWidget(self.render_type_widget)

        self.render_color_markers = rh.add_markers(activate=True, coord_type='relative',
            move_callback=self._render_marker_moved,
            color_change_callback=lambda mrk, cb=self._update_palettes: cb())
        self.render_color_markers.extend([((0.0, 0.0), "blue"), ((0.5, 0.0), "white"), ((1.0, 0.0), "red")])
        self.render_color_markers.add_del_callback = lambda mrk=None, cb=self._update_palettes: cb()
        self.render_markers[self.RENDER_COLORS] = self.render_color_markers
        color_render_tab = QWidget()
        color_render_tab_layout = crt_layout = QVBoxLayout()
        crt_layout.setSpacing(1)
        crt_layout.setContentsMargins(2,2,2,2)
        color_render_tab.setLayout(color_render_tab_layout)
        color_target_layout = QHBoxLayout()
        crt_layout.addLayout(color_target_layout)
        color_target_layout.addStretch(1)
        color_target_layout.addWidget(QLabel("Color: "), alignment=Qt.AlignCenter)
        self.color_atoms = QCheckBox("atoms")
        self.color_atoms.setChecked(True)
        color_target_layout.addStretch(1)
        color_target_layout.addWidget(self.color_atoms, alignment=Qt.AlignCenter)
        self.color_cartoons = QCheckBox("cartoons")
        self.color_cartoons.setChecked(True)
        color_target_layout.addStretch(1)
        color_target_layout.addWidget(self.color_cartoons, alignment=Qt.AlignCenter)
        self.color_surfaces = QCheckBox("surfaces")
        self.color_surfaces.setChecked(True)
        color_target_layout.addStretch(1)
        color_target_layout.addWidget(self.color_surfaces, alignment=Qt.AlignCenter)
        color_target_layout.addStretch(1)
        no_value_layout = QHBoxLayout()
        no_value_layout.addStretch(1)
        self.color_no_value = QCheckBox("Color missing values with:")
        no_value_layout.addWidget(self.color_no_value, alignment=Qt.AlignRight)
        no_value_layout.addSpacing(7)
        from chimerax.ui.widgets import ColorButton
        self.no_value_color = ColorButton(has_alpha_channel=True, max_size=(16,16))
        self.no_value_color.color = "gray"
        no_value_layout.addWidget(self.no_value_color)
        no_value_layout.addStretch(1)
        crt_layout.addLayout(no_value_layout)
        self.palette_chooser = PaletteChooser(self._new_palette)
        crt_layout.addWidget(self.palette_chooser, alignment=Qt.AlignCenter)
        reverse_layout = QHBoxLayout()
        reverse_layout.addStretch(1)
        self.reverse_colors_button = QPushButton("Reverse")
        self.reverse_colors_button.clicked.connect(self._reverse_colors)
        reverse_layout.addWidget(self.reverse_colors_button, alignment=Qt.AlignRight)
        reverse_layout.addWidget(QLabel(" threshold colors"), alignment=Qt.AlignLeft)
        reverse_layout.addStretch(1)
        crt_layout.addLayout(reverse_layout)
        key_layout = QHBoxLayout()
        key_layout.addStretch(1)
        self.key_button = QPushButton("Create")
        self.key_button.clicked.connect(self._create_key)
        key_layout.addWidget(self.key_button, alignment=Qt.AlignRight)
        key_layout.addWidget(QLabel(" corresponding color key"), alignment=Qt.AlignLeft)
        key_layout.addStretch(1)
        crt_layout.addLayout(key_layout)
        self.render_type_widgets[self.RENDER_COLORS] = []
        self.render_type_widget.addTab(color_render_tab, self.RENDER_COLORS)

        self.render_radius_markers = rh.add_markers(new_color='slate gray', activate=False,
            coord_type='relative', add_del_callback=self._radius_marker_add_del)
        for pos, radius in [(0.0, 1.0), (1.0, 4.0)]:
            self.render_radius_markers.append(((pos, 0.0), None)).radius = radius
        self.render_markers[self.RENDER_RADII] = self.render_radius_markers
        from chimerax.ui.options import OptionsPanel, EnumOption, BooleanOption, FloatOption
        radii_render_tab = self.radii_options = OptionsPanel(sorting=False, scrolled=False,
            contents_margins=(0,0,0,0))
        from chimerax.std_commands.size import AtomRadiiStyleArg as radii_arg
        self.radii_style_option = EnumOption("Atom style", radii_arg.default, None,
            values=radii_arg.values)
        self.radii_options.add_option(self.radii_style_option)
        self.radii_affect_nv = BooleanOption("Affect no-value atoms", False, None)
        self.radii_options.add_option(self.radii_affect_nv)
        self.radii_nv_radius = FloatOption("No-value radius", 0.5, None, min="positive")
        self.radii_options.add_option(self.radii_nv_radius)
        radius_label = QLabel("Atom radius")
        rh.add_custom_widget(radius_label, left_side=False, alignment=Qt.AlignRight)
        self.radius_value_entry = QLineEdit()
        validator = QDoubleValidator()
        validator.setBottom(0.001)
        self.radius_value_entry.setValidator(validator)
        from chimerax.ui import set_line_edit_width
        set_line_edit_width(self.radius_value_entry, 5)
        rh.add_custom_widget(self.radius_value_entry, left_side=False, alignment=Qt.AlignLeft)
        rv_widgets = [radius_label, self.radius_value_entry]
        self.render_type_widgets[self.RENDER_RADII] = rv_widgets
        self.render_type_widget.addTab(radii_render_tab, self.RENDER_RADII)

        self.render_worm_markers = rh.add_markers(new_color='slate gray', activate=False,
            coord_type='relative', add_del_callback=self._worms_marker_add_del)
        for pos, radius in [(0.0, 0.25), (1.0, 2.0)]:
            self.render_worm_markers.append(((pos, 0.0), None)).radius = radius
        self.render_markers[self.RENDER_WORMS] = self.render_worm_markers
        worms_render_tab = QWidget()
        worms_render_tab_layout = wrt_layout = QVBoxLayout()
        worms_render_tab.setLayout(wrt_layout)
        self.worms_options = OptionsPanel(sorting=False, scrolled=False, contents_margins=(0,0,0,0))
        wrt_layout.addWidget(self.worms_options, alignment=Qt.AlignCenter)
        self.worm_nv_radius = FloatOption("No-value radius", 0.1, None, min="positive")
        self.worms_options.add_option(self.worm_nv_radius)
        self.deworm_button = deworm_button = QPushButton("Deworm")
        def deworm_cb(*args, self=self):
            models = self.model_list.value
            if not models:
                raise UserError("No models chosen for deworming")
            self._cur_attr_info().render(self.session, None, models, "worm", (False, []),
                self.sel_restrict.isChecked())
            self.deworm_button.setEnabled(False)
        deworm_button.clicked.connect(deworm_cb)
        wrt_layout.addWidget(deworm_button, alignment=Qt.AlignHCenter|Qt.AlignBottom, stretch=1)
        worm_label = QLabel("Worm radius")
        rh.add_custom_widget(worm_label, left_side=False, alignment=Qt.AlignRight)
        self.worm_value_entry = QLineEdit()
        validator = QDoubleValidator()
        validator.setBottom(0.0001)
        self.worm_value_entry.setValidator(validator)
        from chimerax.ui import set_line_edit_width
        set_line_edit_width(self.worm_value_entry, 5)
        rh.add_custom_widget(self.worm_value_entry, left_side=False, alignment=Qt.AlignLeft)
        self.render_type_widgets[self.RENDER_WORMS] = [worm_label, self.worm_value_entry]
        self.render_type_widget.addTab(worms_render_tab, self.RENDER_WORMS)

        self.sel_restrict = QCheckBox()
        self.sel_restrict.setChecked(False)
        self.sel_restrict.toggled.connect(lambda *args, self=self: self._update_deworm_button())
        render_tab_layout.addWidget(self.sel_restrict, alignment=Qt.AlignCenter)
        self._render_mode_changed(self.render_type_widget.currentIndex())
        self.mode_widget.addTab(render_tab, "Render")

        # wait until tab contents are completely filled before connecting this
        self.render_type_widget.currentChanged.connect(self._render_mode_changed)

        sel_tab = QWidget()
        sel_layout = QVBoxLayout()
        sel_tab.setLayout(sel_layout)
        sel_layout.addWidget(QLabel("This tab not yet implemented.\nUse 'select' command instead.",
            alignment=Qt.AlignCenter))
        self.mode_widget.addTab(sel_tab, "Select")

        self._update_target_menu()
        self._new_attr()

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        bbox = qbbox(qbbox.Ok | qbbox.Apply | qbbox.Close | qbbox.Help)
        bbox.accepted.connect(self.render)
        bbox.button(qbbox.Apply).clicked.connect(lambda: self.render(apply=True))
        bbox.rejected.connect(self.delete)
        if hasattr(self, 'help'):
            from chimerax.core.commands import run
            bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help " + self.help))
        else:
            bbox.button(qbbox.Help).setEnabled(False)
        overall_layout.addWidget(bbox)

        tw.manage(placement=None)

    def render(self, *, apply=False):
        models = self.model_list.value
        if not models:
            raise UserError("No models chosen for rendering")
        attr_name = self.attr_menu_button.text()
        if attr_name == self.NO_ATTR_TEXT:
            raise UserError("No attribute chosen for rendering")
        tabs = self.render_type_widget
        tab_text = tabs.tabText(tabs.currentIndex()).lower()
        method = tab_text[:-1] if tab_text[-1] == 's' else "radius"
        markers = getattr(self, "render_" + method + "_markers")
        vals = []
        markers.coord_type = "absolute"
        for marker in markers:
            vals.append((marker.xy[0], getattr(marker, "rgba" if method == "color" else "radius")))
        markers.coord_type = "relative"
        if method == "color":
            targets = set()
            for target in ["atoms", "cartoons", "surfaces"]:
                target_widget = getattr(self, "color_" + target)
                if target_widget.isChecked() and target_widget.isEnabled():
                    targets.add(target)
            if not targets:
                raise UserError("No coloring targets specified")
            # histograms values + possibly None
            if self.color_no_value.isChecked():
                vals.append((None, [v/255.0 for v in self.no_value_color.color]))
            if not vals:
                raise UserError("No coloring values specified")
            params = (targets, vals)
        elif method == "radius":
            if self.radii_affect_nv.widget.isEnabled() and self.radii_affect_nv.value:
                vals.append((None, self.radii_nv_radius.value))
            if not vals:
                raise UserError("No radius values specified")
            params = (self.radii_style_option.value, vals)
        elif method == "worm":
            if not vals:
                raise UserError("No radius values specified")
            if self.worm_nv_radius.widget.isEnabled():
                vals.append((None, self.worm_nv_radius.value))
            params = (True, vals)
        else:
            raise NotImplementedError("Don't know how to get parameters for '%s' method" % tab_text)
        self._cur_attr_info().render(self.session, attr_name, models, method, params,
            self.sel_restrict.isChecked())
        if not apply:
            self.delete()
        elif method == "worm":
            self._update_deworm_button()

    def _attr_names_of_type(self, *types):
        attr_info = self._cur_attr_info()
        from chimerax.core.attributes import MANAGER_NAME
        attr_mgr = self.session.get_state_manager(MANAGER_NAME)
        return [attr_name for attr_name in attr_mgr.attributes_returning(
            attr_info.class_object, types, none_okay=True) if not attr_info.hide_attr(
            attr_name, self.mode_widget.tabText(self.mode_widget.currentIndex()) == "Render")]

    def _create_key(self):
        from chimerax.core.colors import Colormap, Color
        colors = []
        values = []
        crd_type = self.render_color_markers.coord_type
        self.render_color_markers.coord_type = 'absolute'
        for marker in self.render_color_markers:
            colors.append(Color(marker.rgba))
            values.append(marker.xy[0])
        self.render_color_markers.coord_type = crd_type
        cmap = Colormap(values, colors, color_no_value=[x/255.0 for x in self.no_value_color.color])

        from chimerax.color_key import show_key
        show_key(self.session, cmap)

    def _cur_attr_info(self):
        target = self.target_menu_button.text()
        return self._ui_to_info[target]

    def _filter_model(self, model):
        try:
            return self._cur_attr_info().model_filter(model)
        except (AttributeError, KeyError):
            return False

    def _models_changed(self):
        if self.model_list.value and self.attr_menu_button.isEnabled():
            attr_info = self.attr_menu_button.text()
            if attr_info != self.NO_ATTR_TEXT:
                self._update_histogram(attr_info)
        else:
            self._new_attr()
        self._update_deworm_button()

    def _new_attr(self, attr_name_info=None):
        enabled = True
        if attr_name_info is None:
            if not self.model_list.value:
                attr_name = "no model chosen"
                enabled = False
            else:
                attr_name = self.NO_ATTR_TEXT
        else:
            if isinstance(attr_name_info, str):
                attr_name = attr_name_info
            else:
                attr_name = attr_name_info.text()
        if attr_name != self.attr_menu_button.text():
            self.attr_menu_button.setText(attr_name)
            if attr_name_info is None:
                self.render_histogram.data_source = "Choose attribute to show histogram"
            else:
                self._update_histogram(attr_name)
            self._update_palettes()
        self.attr_menu_button.setEnabled(enabled)

    def _new_classes(self):
        self._update_target_menu()

    def _new_palette(self, palette_name):
        for marker, rgba in zip(self.render_color_markers, self.palette_chooser.rgbas):
            marker.rgba = rgba

    def _new_target(self, target):
        if not isinstance(target, str):
            target = target.text()
        self.target_menu_button.setText(target)
        self.model_list.refresh()
        color_targets = self._ui_to_color_targets.get(target, set())
        self.color_atoms.setEnabled("atoms" in color_targets)
        self.color_cartoons.setEnabled("cartoons" in color_targets)
        self.color_surfaces.setEnabled("surfaces" in color_targets)
        self._new_attr()

    def _radius_marker_add_del(self, marker=None):
        if marker:
            marker.radius = 1.0

    def _render_marker_moved(self, move_info):
        if move_info == "end":
            self._update_palettes()

    def _render_mode_changed(self, tab_index):
        render_type = self.render_type_widget.tabText(tab_index)
        self.render_histogram.activate(self.render_markers[render_type])
        self.render_histogram.color_button = render_type == self.RENDER_COLORS
        self.sel_restrict.setText("Restrict to selection"
            if render_type != self.RENDER_WORMS else "Restrict to selected models")
        for category, widgets in self.render_type_widgets.items():
            for widget in widgets:
                widget.setHidden(category != render_type)

    def _render_sel_marker_cb(self, prev_markers, prev_marker, cur_markers, cur_marker):
        if cur_markers == self.render_color_markers:
            return
        value_entry = self.radius_value_entry \
            if cur_markers == self.render_radius_markers else self.worm_value_entry
        if cur_marker:
            value_entry.setText("%g" % cur_marker.radius)
        else:
            value_entry.clear()

    def _reverse_colors(self):
        if len(self.render_color_markers) < 2:
            return
        rgbas = [m.rgba for m in self.render_color_markers]
        rgbas.reverse()
        cb = self.render_color_markers.color_change_callback
        self.render_color_markers.color_change_callback = None
        for m, rgba in zip(self.render_color_markers, rgbas):
            m.rgba = rgba
        self.render_color_markers.color_change_callback = cb
        self._update_palettes()

    def _update_attr_menu(self):
        menu = self.attr_menu_button.menu()
        menu.clear()
        attr_names = self._attr_names_of_type(int, float)
        attr_names.sort()
        for attr_name in attr_names:
            menu.addAction(attr_name)

    def _update_deworm_button(self):
        models = self.model_list.value
        if models and self.sel_restrict.isChecked():
            sel_models = set(self.session.selection.models())
            models = [m for m in models if m in sel_models]
        if models:
            enable = self._cur_attr_info().deworm_applicable(models)
        else:
            enable = False
        self.deworm_button.setEnabled(enable)

    def _update_histogram(self, attr_name):
        attr_info = self._cur_attr_info()
        values, any_None = attr_info.values(attr_name, self.model_list.value)
        if len(values) == 0:
            self.render_histogram.data_source = "No '%s' values for histogram" % attr_name
        else:
            min_val, max_val = min(values), max(values)
            import numpy
            if min_val == max_val:
                self.render_histogram.data_source = "All '%s' values are %g" % (attr_name, min_val)
            elif attr_name in self._attr_names_of_type(int):
                # just histogram the values directly
                self.render_histogram.data_source = (min_val, max_val, numpy.histogram(
                    values, bins=max_val-min_val+1, range=(min_val, max_val), density=False)[0])
            else:
                # number of bins based on histogram pixel width...
                self.render_histogram.data_source = (min_val, max_val, lambda num_bins:
                    numpy.histogram(values, bins=num_bins, range=(min_val, max_val), density=False)[0])
        self.radii_options.set_option_enabled(self.radii_affect_nv, not any_None)
        self.radii_options.set_option_enabled(self.radii_nv_radius, not any_None)
        self.worms_options.set_option_enabled(self.worm_nv_radius, not any_None)

    def _update_palettes(self):
        if type(self.render_histogram.data_source) == str:
            rgbas = []
        else:
            rgbas = [m.rgba for m in self.render_color_markers]
        self.reverse_colors_button.setEnabled(len(rgbas) > 1)
        self.key_button.setEnabled(len(rgbas) > 1)
        self.palette_chooser.rgbas = rgbas

    def _update_target_menu(self):
        from .manager import get_manager
        mgr = get_manager(self.session)
        self._ui_to_info = {}
        self._ui_to_color_targets = {}
        ui_names = []
        for pn in mgr.provider_names:
            ui_name = mgr.ui_name(pn)
            ui_names.append(ui_name)
            self._ui_to_info[ui_name] = mgr.render_attr_info(pn)
            self._ui_to_color_targets[ui_name] = mgr.color_targets(pn)
        ui_names.sort()
        menu = self.target_menu_button.menu()
        menu.clear()
        for ui_name in ui_names:
            menu.addAction(ui_name)
        if not self.target_menu_button.text() and ui_names:
            self._new_target(ui_names[0])

    def _worms_marker_add_del(self, marker=None):
        if marker:
            marker.radius = 0.25

