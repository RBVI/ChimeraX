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
    from Qt.QtGui import QAction
    plot_menu = menu.addMenu("Plot")
    action = QAction("Distances", plot_menu)
    action.triggered.connect(lambda *args, tw=parent_tool_window, s=structure: _show_distances_plot(tw, s))
    plot_menu.addAction(action)

class PlotDialog:
    def __init__(self, plot_window, structure):
        self.tool_window = tw = plot_window
        self.session = structure.session
        from Qt.QtWidgets import QHBoxLayout
        self.section_layout = layout = QVBoxLayout()
        layout.setSpacing(0)
        tw.ui_area.setLayout(layout)

        self.sections = {}

        tw.manage(None)

    def make_section(self, section_name):
        from Qt.QtWidgets import QWidget, QLabel, QHBoxLayout
        section = QWidget()
        section_layout = QHBoxLayout()
        section_layout.setSpacing(0)
        section.setLayout(section_layout)
        self.section_layout.addWidget(section, stretch=1)
        if section_name == "distances":
            section_layout.addWidget(QLabel("Distance plotting goes here"))
        else:
            raise ValueError("Don't know how to make plot section '%s'" % section_name)
        return section

    def show_section(self, section_name):
        try:
            section = self.sections[section_name]
        except KeyError:
            section = self.sections[section_name] = self.make_section(section_name)

        section.show()

def _show_distances_plot(main_tool_window, structure):
    try:
        tws = main_tool_window._md_tool_windows
    except AttributeError:
        tws = main_tool_window._md_tool_windows = {}

    try:
        plot_dialog = tws["plot"]
    except KeyError:
        plot_dialog = tws["plot"] = PlotDialog(main_tool_window.create_child_window("Plots"), structure)

    plot_dialog.show_section("distances")
    plot_dialog.tool_window.shown = True

