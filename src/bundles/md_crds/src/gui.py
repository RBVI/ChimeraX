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

from Qt.QtWidgets import QFrame, QVBoxLayout, QLabel, QHBoxLayout, QCheckBox, QPushButton, QMenu, \
    QGridLayout, QSizePolicy
from Qt.QtCore import Qt

class SaveOptionsWidget(QFrame):

    def __init__(self, session):
        super().__init__()
        self.session = session

        layout = QVBoxLayout()
        layout.setContentsMargins(2, 0, 0, 0)
        layout.setSpacing(5)

        models_layout = QVBoxLayout()
        layout.addLayout(models_layout, stretch=1)
        models_layout.setSpacing(0)
        models_label = QLabel("Save models")
        from chimerax.ui import shrink_font
        shrink_font(models_label)
        models_layout.addWidget(models_label, alignment=Qt.AlignLeft)
        from chimerax.atomic.widgets import StructureListWidget
        self.structure_list = StructureListWidget(session)
        self.structure_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        models_layout.addWidget(self.structure_list)

        self.setLayout(layout)

    def options_string(self):
        models = self.structure_list.value
        from chimerax.core.errors import UserError
        if not models:
            raise UserError("No models chosen for saving")
        from chimerax.atomic import Structure
        from chimerax.core.commands import concise_model_spec
        spec = concise_model_spec(self.session, models, relevant_types=Structure)
        if spec:
            cmd = "models " + spec
        else:
            cmd = ""
        return cmd

def fill_context_menu(menu, parent_tool_window, structure):
    from .manager import get_plotting_manager
    mgr = get_plotting_manager(structure.session)

    from Qt.QtGui import QAction
    plot_menu = menu.addMenu("Plot")

    from chimerax.core.commands import plural_of
    for provider_name in mgr.provider_names:
        ui_name = mgr.ui_name(provider_name)
        menu_name = plural_of(ui_name)
        if menu_name.lower() == menu_name:
            # no caps
            menu_name = ui_name.capitalize()

        action = QAction(menu_name, plot_menu)
        action.triggered.connect(lambda *args, tw=parent_tool_window, s=structure, name=provider_name:
            _show_plot(name, tw, s))
        plot_menu.addAction(action)

class PlotDialog:
    def __init__(self, plot_window, structure):
        self.tool_window = tw = plot_window
        self.session = structure.session
        from .manager import get_plotting_manager
        self.mgr = get_plotting_manager(self.session)
        from Qt.QtWidgets import QHBoxLayout, QTabWidget
        layout = QVBoxLayout()
        layout.setSpacing(0)
        tw.ui_area.setLayout(layout)
        self.plot_tabs = QTabWidget()
        self.plot_tabs.setTabsClosable(True)
        #TODO tabCloseRequested(index) signal
        layout.addWidget(self.plot_tabs, stretch=1)

        self.tab_info = {}
        self._tables = {}

        tw.manage(None)

    def make_tab(self, provider_name):
        if self.mgr.num_atoms(provider_name) == 0:
            return self._make_scalar_tab(provider_name)
        return self._make_atomic_tab(provider_name)

    def show_tab(self, provider_name):
        try:
            tab_name, tab_widget = self.tab_info[provider_name]
        except KeyError:
            tab_name, tab_widget = self.tab_info[provider_name] = self.make_tab(provider_name)

        self.plot_tabs.setCurrentWidget(tab_widget)

    def _make_atomic_tab(self, provider_name):
        ui_name = self.mgr.ui_name(provider_name)
        from chimerax.core.commands import plural_of
        tab_name = plural_of(ui_name)
        if tab_name.lower() == tab_name:
            # no caps
            tab_name = tab_name.capitalize()
        from Qt.QtWidgets import QWidget, QLabel, QHBoxLayout, QVBoxLayout
        page = QWidget()
        page_layout = QHBoxLayout()
        page_layout.setSpacing(0)
        page.setLayout(page_layout)
        page_layout.addWidget(QLabel("%s plotting goes here" % ui_name.capitalize()), stretch=1)
        controls_area = QWidget()
        controls_layout = QVBoxLayout()
        controls_area.setLayout(controls_layout)
        self._tables[tab_name] = table = self._make_table(provider_name)
        controls_layout.addWidget(table, stretch=1)
        controls_layout.addWidget(QLabel("Controls"))
        page_layout.addWidget(controls_area)
        self.plot_tabs.addTab(page, tab_name)
        return tab_name, page

    def _make_scalar_tab(self, provider_name):
        #TODO
        raise NotImplementedError("Scalar plotting not implemented")

    def _make_table(self, provider_name):
        from chimerax.ui.widgets import ItemTable
        #TODO
        from Qt.QtWidgets import QLabel
        return QLabel("Table")

def _show_plot(provider_name, main_tool_window, structure):
    try:
        tws = main_tool_window._md_tool_windows
    except AttributeError:
        tws = main_tool_window._md_tool_windows = {}

    try:
        plot_dialog = tws["plot"]
    except KeyError:
        plot_dialog = tws["plot"] = PlotDialog(main_tool_window.create_child_window("MD Plots"), structure)

    plot_dialog.show_tab(provider_name)
    plot_dialog.tool_window.shown = True

