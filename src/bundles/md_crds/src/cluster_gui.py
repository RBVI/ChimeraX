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

class ClusterLauncher:

    def __init__(self, launcher_window, structure):
        self.tool_window = tw = launcher_window
        #tw.help = "help:user/commands/coordset.html#slider"
        def cleanup(lcd=self):
            inst = lcd.tool_window.tool_instance
            from .gui import _remove_tool_window
            _remove_tool_window(inst, "cluster launcher")
            delattr(lcd.tool_window, 'cleanup')
        tw.cleanup = cleanup
        self.session = structure.session
        self.structure = structure
        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0,0,0,0)
        tw.ui_area.setLayout(layout)

        from chimerax.ui.options import OptionsPanel, IntOption, BooleanOption, EnumOption
        options_panel = OptionsPanel(sorting=False, scrolled=False)
        cs_ids = structure.coordset_ids
        min_cs = min(cs_ids)
        max_cs = max(cs_ids)
        self.start_opt = IntOption("Starting frame:", min_cs, None, min=min_cs, max=max_cs)
        options_panel.add_option(self.start_opt)
        self.step_opt = IntOption("Step size:", 1 + int(len(cs_ids)/300), None, min=1, max=max_cs)
        options_panel.add_option(self.step_opt)
        self.end_opt = IntOption("Ending frame:", max_cs, None, min=min_cs, max=max_cs)
        options_panel.add_option(self.end_opt)
        self.sel_opt = BooleanOption("Cluster based on current selection, if any:", True, None)
        options_panel.add_option(self.sel_opt)
        self.solvent_opt = BooleanOption("Ignore solvent and non-metal ions:", True, None)
        options_panel.add_option(self.solvent_opt)
        self.hyd_opt = BooleanOption("Ignore hydrogens:", True, None)
        options_panel.add_option(self.hyd_opt)
        self.ligand_opt = BooleanOption("Ignore ligands:", False, None)
        options_panel.add_option(self.ligand_opt)
        from .manager import get_plotting_manager
        mgr = get_plotting_manager(self.session)
        self.metal_opt = EnumOption("Ignore metal ions:", "alkali", None, values=mgr.exclude_info["metals"])
        options_panel.add_option(self.metal_opt)
        layout.addWidget(options_panel)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        self.bbox = bbox = qbbox(qbbox.Ok | qbbox.Apply | qbbox.Close | qbbox.Help)
        bbox.accepted.connect(self.launch_clustering)
        bbox.button(qbbox.Apply).clicked.connect(lambda *args: self.launch_clustering(apply=True))
        bbox.rejected.connect(tw.destroy)
        if getattr(tw, 'help', None):
            from chimerax.core.commands import run
            bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help " + tw.help))
        else:
            bbox.button(qbbox.Help).setEnabled(False)
        layout.addWidget(bbox)

        tw.manage(None)

    def launch_clustering(self, *, apply=False):
        start = self.start_opt.value
        step = self.step_opt.value
        end = self.end_opt.value
        sel = self.sel_opt.value
        solvent = self.solvent_opt.value
        hyd = self.hyd_opt.value
        ligand = self.ligand_opt.value
        metal = self.metal_opt.value
        if not apply:
            self.tool_window.destroy()
        from chimerax.core.commands import run
        spec = '#!' + self.structure.id_string
        if sel and self.structure.atoms.selecteds.any():
            spec += " & sel"
        cmd = f"md cluster {spec} start {start} step {step} end {end}"
        if not solvent:
            cmd += " excludeSolvent false"
        if not hyd:
            cmd += " excludeHydrogens false"
        if ligand:
            cmd += " excludeLigands true"
        if metal != "alkali":
            cmd += " excludeMetals " + metal
        run(self.session, cmd)

def _show_cluster_launcher(main_tool_window, structure):
    inst = main_tool_window.tool_instance
    inst_windows = _md_tool_windows.setdefault(inst, {})
    try:
        cluster_launcher = inst_windows["cluster launcher"]
    except KeyError:
        cluster_launcher = inst_windows["cluster launcher"] = ClusterLauncher(
            main_tool_window.create_child_window("Get Clustering Parameters"), structure)

    cluster_launcher.tool_window.shown = True

