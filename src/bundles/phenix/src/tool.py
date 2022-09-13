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
from chimerax.core.settings import Settings
from Qt.QtCore import Qt

class DouseSettings(Settings):
    AUTO_SAVE = {
        "show_hbonds": True,
    }

from chimerax.check_waters.tool import CheckWaterViewer
class DouseResultsViewer(CheckWaterViewer):
    def __init__(self, session, tool_name, orig_model=None, douse_model=None, compared_waters=None,
            map=None):
        # if 'model' is None, we are being restored from a session and _finalize_init() will be called later
        super().__init__(session, tool_name, douse_model, compare_info=(orig_model, compared_waters),
            model_labels=("input", "douse"), compare_map=map, category_tips=(
            "Waters placed by douse that are not in input model",
            "Waters in input model and also found by douse",
            "Waters in input model only"
            ))

from chimerax.core.settings import Settings
from .douse import command_defaults as defaults
class LaunchDouseSettings(Settings):
    AUTO_SAVE = {
        'first_shell': not defaults['far_water'],
        'keep_waters': defaults['keep_input_water'],
        'hide_map': defaults['map_range'] > 0,
        'hide_map_dist': defaults['map_range'],
        'res_range': defaults['residue_range'],
        'verbose': defaults['verbose'],
    }

class LaunchDouseTool(ToolInstance):
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
        options = OptionsPanel(scrolled=False)
        layout.addWidget(options)
        self.first_shell_option = BooleanOption("First shell only", None, None,
            attr_name="first_shell", settings=self.settings,
            balloon="Only place waters that interact directly with structure, rather than other waters")
        options.add_option(self.first_shell_option)
        self.keep_waters_option = BooleanOption("Retain waters from input structure in generated structure",
            None, None, attr_name="keep_waters", settings=self.settings)
        options.add_option(self.keep_waters_option)
        self.hide_map_option = BooleanOption("Hide map far from waters", None, None,
            attr_name="hide_map", settings=self.settings)
        options.add_option(self.hide_map_option)
        self.hide_map_dist_option = FloatOption("Map-hiding distance (angstroms)", None, None,
            attr_name="hide_map_dist", settings=self.settings, decimal_places=1, step=1, min=0, max=99)
        options.add_option(self.hide_map_dist_option)
        self.res_range_option = FloatOption("Show residues with atoms within this distance of waters", None,
            None, attr_name="res_range", settings=self.settings, decimal_places=1, step=1, min=0, max=99)
        options.add_option(self.res_range_option)
        self.verbose_option = BooleanOption("Show full douse output in log", None, None,
            attr_name="verbose", settings=self.settings)
        options.add_option(self.verbose_option)
        from .douse import douse_needs_resolution
        if douse_needs_resolution(session):
            self.resolution_option = FloatOption("Map resolution", 3.0, None, min=0.0, max=9999.9)
            options.add_option(self.resolution_option)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        self.bbox = bbox = qbbox(qbbox.Ok | qbbox.Close | qbbox.Help)
        bbox.accepted.connect(self.launch_douse)
        bbox.accepted.connect(self.delete) # slots executed in the order they are connected
        bbox.rejected.connect(self.delete)
        #from chimerax.core.commands import run
        #bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help " + self.help))
        bbox.button(qbbox.Help).setEnabled(False)
        layout.addWidget(bbox)

        tw.manage(placement=None)

    def launch_douse(self):
        structure = self.structure_menu.value
        if not structure:
            # raising UserError leaves C++ destroyed, but not Python
            self.session.logger.error("Must specify a structure for water placement")
            return
        map = self.map_menu.value
        if not map:
            # raising UserError leaves C++ destroyed, but not Python
            self.session.logger.error("Must specify a map for water placement")
            return
        cmd = "phenix douse %s near %s" % (map.atomspec, structure.atomspec)
        from chimerax.core.commands import BoolArg
        first_shell = self.first_shell_option.value
        if first_shell != (not defaults['far_water']):
            cmd +=  " farWater %s" % BoolArg.unparse(not first_shell)
        keep_waters = self.keep_waters_option.value
        if keep_waters != defaults['keep_input_water']:
            cmd += " keepInputWater %s" % BoolArg.unparse(keep_waters)
        hide_map = self.hide_map_option.value
        if hide_map:
            hide_map_dist = self.hide_map_dist_option.value
            if hide_map_dist != defaults['map_range']:
                cmd += " mapRange %g" % hide_map_dist
        elif defaults['map_range'] > 0:
            cmd += " mapRange 0"
        res_range = self.res_range_option.value
        if res_range != defaults['residue_range']:
            cmd += " residueRange %g" % res_range
        verbose = self.verbose_option.value
        if verbose != defaults['verbose']:
            cmd += " verbose %s" % BoolArg.unparse(verbose)
        if hasattr(self, 'resolution_option'):
            cmd += " resolution %g" % self.resolution_option.value
        from chimerax.core.commands import run
        run(self.session, cmd)
