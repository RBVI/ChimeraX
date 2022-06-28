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

from Qt.QtCore import Qt

class RenderByAttrTool(ToolInstance):

    #help = "help:user/tools/matchmaker.html"

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)
        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self, statusbar=True)
        parent = tw.ui_area
        from Qt.QtWidgets import QVBoxLayout, QHBoxLayout, QDialogButtonBox, QPushButton, QMenu, QLabel
        from Qt.QtWidgets import QTabWidget, QWidget
        from Qt.QtCore import Qt
        overall_layout = QVBoxLayout()
        overall_layout.setContentsMargins(0,0,0,0)
        overall_layout.setSpacing(0)
        parent.setLayout(overall_layout)

        target_layout = QHBoxLayout()
        overall_layout.addLayout(target_layout)

        target_layout.addWidget(QLabel("Attributes of"), alignment=Qt.AlignRight)
        self.target_menu_button = QPushButton()
        menu = QMenu()
        menu.triggered.connect(self._new_target)
        self.target_menu_button.setMenu(menu)
        target_layout.addWidget(self.target_menu_button, alignment=Qt.AlignLeft)
        model_list_layout = QVBoxLayout()
        target_layout.addLayout(model_list_layout)
        model_list_layout.addWidget(QLabel("Models"), alignment=Qt.AlignBottom)
        from chimerax.ui.widgets import ModelListWidget, MarkedHistogram
        class ShortModelListWidget(ModelListWidget):
            def sizeHint(self):
                sh = super().sizeHint()
                sh.setHeight(sh.height() // 2)
                return sh
        self.model_list = ShortModelListWidget(session, filter_func=self._filter_model)
        self.model_list.value_changed.connect(self._models_changed)
        model_list_layout.addWidget(self.model_list, alignment=Qt.AlignTop)

        self.mode_widget = QTabWidget()
        overall_layout.addWidget(self.mode_widget)

        render_tab = QWidget()
        render_tab_layout = QVBoxLayout()
        render_tab.setLayout(render_tab_layout)
        attr_menu_layout = QHBoxLayout()
        render_tab_layout.addLayout(attr_menu_layout)
        attr_menu_layout.addWidget(QLabel("Attribute:"), alignment=Qt.AlignRight)
        self.attr_menu_button = QPushButton()
        menu = QMenu()
        menu.triggered.connect(self._new_render_attr)
        self.attr_menu_button.setMenu(menu)
        attr_menu_layout.addWidget(self.attr_menu_button, alignment=Qt.AlignLeft)
        self.render_histogram = rh = MarkedHistogram(min_label=True, max_label=True, status_line=tw.status)
        render_tab_layout.addWidget(rh)
        self.render_color_markers = rh.add_markers(activate=True, coord_type='relative')
        self.render_color_markers.extend([((0.0, 0.0), "red"), ((0.5, 0.0), "white"), ((1.0, 0.0), "blue")])
        self.render_radii_markers = rh.add_markers(new_color='slate gray', activate=False,
            coord_type='relative')
        self.render_radii_markers.extend([((0.0, 0.0), None), ((1.0, 0.0), None)])
        self.mode_widget.addTab(render_tab, "Render")

        sel_tab = QWidget()
        sel_layout = QVBoxLayout()
        sel_tab.setLayout(sel_layout)
        sel_layout.addWidget(QLabel("This tab not yet implemented.\nUse 'select' command instead.",
            alignment=Qt.AlignCenter))
        self.mode_widget.addTab(sel_tab, "Select")

        self._update_target_menu()

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        bbox = qbbox(qbbox.Ok | qbbox.Apply | qbbox.Close | qbbox.Help)
        bbox.accepted.connect(self.render)
        bbox.button(qbbox.Apply).clicked.connect(self.render)
        bbox.accepted.connect(self.delete) # slots executed in the order they are connected
        bbox.rejected.connect(self.delete)
        #from chimerax.core.commands import run
        #bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help " + self.help))
        bbox.button(qbbox.Help).setEnabled(False)
        overall_layout.addWidget(bbox)

        tw.manage(placement=None)

    def render(self):
        pass

    def _attr_names_of_type(self, *types):
        attr_info = self._cur_attr_info()
        from chimerax.core.attributes import MANAGER_NAME
        attr_mgr = self.session.get_state_manager(MANAGER_NAME)
        return [attr_name for attr_name in attr_mgr.attributes_returning(
            attr_info.class_object, types, none_okay=True) if not attr_info.hide_attr(
            attr_name, self.mode_widget.tabText(self.mode_widget.currentIndex()) == "Render")]

    def _cur_attr_info(self):
        target = self.target_menu_button.text()
        return self._ui_to_info[target]

    def _filter_model(self, model):
        try:
            return self._ui_to_info[self.target_menu_button.text()].model_filter(model)
        except (AttributeError, KeyError):
            return False

    def _models_changed(self):
        if self.model_list.value and self.attr_menu_button.isEnabled():
            attr_info = self.attr_menu_button.text()
            if attr_info != "choose attr":
                self._update_histogram(attr_info)
        else:
            self._new_render_attr()

    def _new_render_attr(self, attr_name_info=None):
        enabled = True
        if attr_name_info is None:
            if not self.model_list.value:
                attr_name = "no model chosen"
                enabled = False
            else:
                attr_name = "choose attr"
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
        self.attr_menu_button.setEnabled(enabled)


    def _new_classes(self):
        self._update_target_menu()

    def _new_target(self, target):
        if not isinstance(target, str):
            target = target.text()
        self.target_menu_button.setText(target)
        self.model_list.refresh()
        self._update_render_attr_menu()

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

    def _update_render_attr_menu(self, call_new_attr=True):
        menu = self.attr_menu_button.menu()
        menu.clear()
        attr_names = self._attr_names_of_type(int, float)
        attr_names.sort()
        for attr_name in attr_names:
            menu.addAction(attr_name)
        if call_new_attr:
            self._new_render_attr()

    def _update_target_menu(self):
        from .manager import get_manager
        mgr = get_manager(self.session)
        self._ui_to_info = {}
        ui_names = []
        for pn in mgr.provider_names:
            ui_name = mgr.ui_name(pn)
            ui_names.append(ui_name)
            self._ui_to_info[ui_name] = mgr.render_attr_info(pn)
        ui_names.sort()
        menu = self.target_menu_button.menu()
        menu.clear()
        for ui_name in ui_names:
            menu.addAction(ui_name)
        if not self.target_menu_button.text() and ui_names:
            self._new_target(ui_names[0])

