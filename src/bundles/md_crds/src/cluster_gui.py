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

from .gui import _md_tool_windows

class LaunchClusteringDialog:

    def __init__(self, plot_window, structure):
        self.tool_window = tw = plot_window
        #tw.help = "help:user/commands/coordset.html#slider"
        def cleanup(lcd=self):
            inst = lcd.tool_window.tool_instance
            from .gui import _remove_tool_window
            _remove_tool_window(inst, "cluster launcher")
            delattr(lcd.tool_window, 'cleanup')
        tw.cleanup = cleanup
        self.session = structure.session
        self.structure = structure
        from Qt.QtWidgets import QHBoxLayout, QTabWidget
        layout = QVBoxLayout()
        layout.setSpacing(0)
        tw.ui_area.setLayout(layout)

        tw.manage(None)

def _show_cluster_launcher(main_tool_window, structure):
    inst = main_tool_window.tool_instance
    inst_windows = _md_tool_windows.setdefault(inst, {})
    try:
        cluster_launcher = inst_windows["cluster launcher"]
    except KeyError:
        cluster_launcher = inst_windows["cluster launcher"] = ClusterLauncher(
            main_tool_window.create_child_window("Get Clustering Parameters"), structure)

    cluster_launcher.tool_window.shown = True

