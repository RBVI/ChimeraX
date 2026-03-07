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
    QSizePolicy, QWidget, QStackedWidget, QGridLayout, QLineEdit
from Qt.QtGui import QIntValidator
from Qt.QtCore import Qt

from chimerax.core.commands import plural_of
from chimerax.core.errors import UserError

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
    plot_action = menu.addAction("Plot")
    from .plot_gui import show_plot_dialog
    plot_action.triggered.connect(lambda *args, tw=parent_tool_window, s=structure: show_plot_dialog(tw, s))
    cluster_action = menu.addAction("Cluster Frames")
    from .cluster_gui import show_cluster_launcher
    cluster_action.triggered.connect(lambda *args, tw=parent_tool_window, s=structure:
        show_cluster_launcher(tw, s))

_md_tool_windows = {}

def _remove_tool_window(tool_instance, window_type):
    del _md_tool_windows[tool_instance][window_type]
    if not _md_tool_windows[tool_instance]:
        del _md_tool_windows[tool_instance]
