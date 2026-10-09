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
    rmsd_map_action = menu.addAction("RMSD Map")
    from .rmsd_map_gui import show_rmsd_map_launcher
    rmsd_map_action.triggered.connect(lambda *args, tw=parent_tool_window, s=structure:
        show_rmsd_map_launcher(tw, s))

def get_session_info(tool_window):
    data = {}
    from .plot_gui import plot_session_info
    plot_data = plot_session_info(tool_window)
    if plot_data:
        data["plot"] = plot_data
    from .cluster_gui import cluster_dialog_session_info
    cluster_data = cluster_dialog_session_info(tool_window)
    if cluster_data:
        data["cluster"] = cluster_data
    return data

def restore_session_info(parent_tool_window, info):
    if "plot" in info:
        from .plot_gui import restore_plot_info
        restore_plot_info(parent_tool_window, info["plot"])
    if "cluster" in info:
        from .cluster_gui import restore_cluster_info
        restore_cluster_info(parent_tool_window, info["cluster"])

_md_tool_windows = {}

def _remove_tool_window(tool_instance, window_type):
    del _md_tool_windows[tool_instance][window_type]
    if not _md_tool_windows[tool_instance]:
        del _md_tool_windows[tool_instance]

from chimerax.core.settings import Settings
class SaveMatplotImageDialogSettings(Settings):
    AUTO_SAVE = {
        "dpi": None,
        "save_format": "PNG",
        "transparent_background": False,
    }

# Cribbed from chimerax.ui.open_save.SaveDialog, but since we need to save the formats ourselves and
# save some formats otherwise unknown to ChimeraX (e.g. EPS, SVG), we provide our own dialog
from Qt.QtWidgets import QFileDialog
class SaveMatplotImageDialog(QFileDialog):
    def __init__(self, session, parent = None, *args, **kw):
        self.format_info = [
            ("PNG", "Portable Network Graphics", "png"),
            ("JPEG/JPG", "Joint Photographic Experts Group", "jpg *.jpeg"),
            ("TIFF", "Tagged Image File Format", "tiff"),
            ("PDF", "Portable Document Format", "pdf"),
            ("SVG", "Scalable Vector Graphics", "svg"),
            ("EPS", "Encapsulated PostScript", "eps"),
            ("PS", "PostScript", "ps"),
        ]
        name_filters = ["%s [%s] (*.%s)" % fmt_info for fmt_info in self.format_info]
        self.filter_to_info = {flt: info for flt, info in zip(name_filters, self.format_info)}
        fmt_to_filter = { info[0]: flt for flt, info in self.filter_to_info.items() }
        super().__init__(parent, *args, **kw)
        self.setFileMode(QFileDialog.AnyFile)
        self.setAcceptMode(QFileDialog.AcceptSave)
        self.setOption(QFileDialog.DontUseNativeDialog)
        self.setNameFilters(name_filters)
        self.settings = SaveMatplotImageDialogSettings(session, "MD save image dialog")
        try:
            self.selectNameFilter(fmt_to_filter[self.settings.save_format])
        except KeyError:
            self.selectNameFilter(fmt_to_filter["PNG"])

        custom_area = QFrame(self)
        custom_area.setFrameStyle(QFrame.Panel | QFrame.Raised)
        custom_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = self.layout()
        row = layout.rowCount()
        layout.addWidget(custom_area, row, 0, 1, -1)
        custom_layout = QHBoxLayout()
        custom_area.setLayout(custom_layout)
        custom_layout.addStretch(1)
        self._transparent_checkbox = QCheckBox("Transparent background")
        self._transparent_checkbox.setChecked(self.settings.transparent_background)
        custom_layout.addWidget(self._transparent_checkbox)
        custom_layout.addStretch(1)
        custom_layout.addWidget(QLabel("DPI:"))
        self._dpi_entry = QLineEdit()
        self._dpi_entry.setAlignment(Qt.AlignCenter)
        self._dpi_entry.setPlaceholderText("default")
        self._dpi_entry.setMaximumWidth(50)
        validator = QIntValidator()
        validator.setBottom(1)
        self._dpi_entry.setValidator(validator)
        if self.settings.dpi is not None:
            self._dpi_entry.setText(str(self.settings.dpi))
        custom_layout.addWidget(self._dpi_entry)
        custom_layout.addStretch(1)

    @property
    def dpi(self):
        if self._dpi_entry.hasAcceptableInput():
            return int(self._dpi_entry.text())
        return None

    @property
    def path(self):
        paths = self.selectedFiles()
        if not paths:
            return None
        path = paths[0]
        name_filter = self.selectedNameFilter()
        fmt_name, fmt_desc, suffix_info = self.filter_to_info[name_filter]
        self.settings.save_format = fmt_name
        self.settings.transparent_background = self.transparent_background
        self.settings.dpi = self.dpi
        suffix = '.' + (suffix_info[:suffix_info.index(' ')] if ' ' in suffix_info else suffix_info)
        if path.endswith(suffix):
            return path
        return path + suffix

    @property
    def transparent_background(self):
        return self._transparent_checkbox.isChecked()
