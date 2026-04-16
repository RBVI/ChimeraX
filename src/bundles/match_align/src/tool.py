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

_ma_settings = None

import inspect
from .cmd import make_alignment
cmd_params = inspect.signature(make_alignment).parameters
cmd_defaults = {
    kw_name: cmd_params[kw_name].default for kw_name in
        ['cutoff_distance', 'column_criterion', 'gap_char', 'circular', 'max_iterations']
}

class MatchAlignTool(ToolInstance):

    #help = "help:user/tools/findcavities.html"
    SESSION_SAVE = False

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)
        global _ma_settings
        if _ma_settings is None:
            from chimerax.core.settings import Settings
            class _MatchAlignSettings(Settings):
                EXPLICIT_SAVE = cmd_defaults
            _ma_settings = _MatchAlignSettings(self.session, "Match->Align")
        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        tw.title = "Create Alignment from Superposition"
        parent = tw.ui_area
        from Qt.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QWidget, QGroupBox, QCheckBox
        from Qt.QtWidgets import QDoubleSpinBox
        from Qt.QtCore import Qt
        self.layout = layout = QVBoxLayout()
        parent.setLayout(layout)
        layout.setContentsMargins(0,0,0,0)

        from chimerax.atomic.widgets import ChainListWidget
        self.chain_list = ChainListWidget(session, selection_mode="multi")
        layout.addWidget(self.chain_list)
        seen = set()
        initial_chains = []
        for chain in self.chain_list.all_values:
            if chain.structure not in seen:
               initial_chains.append(chain)
               seen.add(chain.structure)
        self.chain_list.value = initial_chains
        self.chain_list.value_changed.connect(self._chains_changed)
        self._prev_chains = { chain.structure: chain for chain in self.chain_list.value }
        '''
        structures_widget = QWidget()
        structures_layout = QHBoxLayout()
        structures_widget.setLayout(structures_layout)
        layout.addWidget(structures_widget, alignment=Qt.AlignCenter)
        structures_layout.addWidget(QLabel("Find cavities in:"), alignment=Qt.AlignRight)
        from chimerax.atomic.widgets import AtomicStructureListWidget
        class ShortASLWidget(AtomicStructureListWidget):
            def sizeHint(self):
                hint = super().sizeHint()
                hint.setHeight(hint.height()//2)
                return hint
        self.structures_list = ShortASLWidget(session, autoselect=ShortASLWidget.AUTOSELECT_SINGLE)
        structures_layout.addWidget(self.structures_list, alignment=Qt.AlignRight)


        group = QGroupBox("Cavity detection settings")
        layout.addWidget(group, alignment=Qt.AlignTop|Qt.AlignHCenter)
        group_layout = QHBoxLayout()
        group_layout.setContentsMargins(0,0,0,0)
        group.setLayout(group_layout)
        from chimerax.ui.options import SettingsPanel, FloatOption
        self.options_panel = panel = SettingsPanel(sorting=False, scrolled=False)
        group_layout.addWidget(panel)
        tool_tips = {
            'probe_in':
                "A smaller probe that defines the biomolecular surface by rolling around\n"
                " the target biomolecule. Typically, this is set to the size of a water\n"
                " molecule (1.4 Å).",
            'probe_out':
                "A larger probe that defines inacessibility region, i.e., the cavities,\n"
                " and by rolling around the target biomolecule. Users can adjust the size\n"
                " of the probe based on the characteristics of the target structure.",
            'removal_distance':
                "A length that is removed from the boundary between the cavity and bulk\n"
                " (solvent) region.",
            'volume_cutoff':
                "A cavity volume filter to exclude cavities with smaller volumes than this\n"
                " limit. These smaller cavities are typically not relevant for function."
        }
        # some of the min/max values are there to make the entry areas less wide
        for label, attr_name, kw in [
                ("Grid spacing", 'grid_spacing', {'min': 'positive'}),
                ("Inner probe radius", 'probe_in', {'min': 'positive'}),
                ("Outer probe radius", 'probe_out', {'min': 'positive'}),
                ("Exterior trim distance", 'removal_distance', { 'min': -999.9}),
                ("Minimum cavity volume", 'volume_cutoff', {'min': 0.0})]:
            opt = FloatOption(label, getattr(_launch_settings, attr_name), None, decimal_places=2,
                balloon=tool_tips.get(attr_name, None), attr_name=attr_name, settings=_launch_settings,
                    max=1000.0, **kw)
            setattr(self, attr_name + '_option', opt)
            panel.add_option(opt)

        restrict_layout = QHBoxLayout()
        restrict_layout.setContentsMargins(0,0,0,0)
        restrict_layout.setSpacing(0)
        restrict_layout.addStretch(1)
        self.restrict_box = QCheckBox("Restrict search to box around selected atoms with padding ")
        self.restrict_box.toggled.connect(lambda enabled, s=self: s.padding_box.setEnabled(enabled))
        restrict_layout.addWidget(self.restrict_box)
        self.padding_box = QDoubleSpinBox()
        self.padding_box.setRange(0, 999.9)
        self.padding_box.setSingleStep(0.5)
        self.padding_box.setDecimals(1)
        self.padding_box.setAlignment(Qt.AlignCenter)
        self.padding_box.setValue(2)
        self.padding_box.setEnabled(False)
        restrict_layout.addWidget(self.padding_box)
        restrict_layout.addStretch(1)
        layout.addLayout(restrict_layout)

        self.include_box = QCheckBox("Include selected atoms as part of macromolecule")
        layout.addWidget(self.include_box, alignment=Qt.AlignCenter)

        self.replace_prev = QCheckBox("Replace existing results, if any")
        self.replace_prev.setChecked(True)
        layout.addWidget(self.replace_prev, alignment=Qt.AlignCenter)
        '''

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        self.bbox = bbox = qbbox(qbbox.Ok | qbbox.Apply | qbbox.Close | qbbox.Help)
        bbox.accepted.connect(self.match_align)
        # Since ApplyRole is not AcceptRole, simply connecting to the Apply button won't dismiss the dialog
        bbox.button(qbbox.Apply).clicked.connect(lambda *args, ma=self.match_align: ma(apply=True))
        bbox.rejected.connect(self.delete)
        if getattr(self, 'help', None) is None:
            bbox.button(qbbox.Help).setEnabled(False)
        else:
            from chimerax.core.commands import run
            bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help " + self.help))
        layout.addWidget(bbox)

        tw.manage(placement=None)

    def match_align(self, apply=False):
        if not apply:
            self.tool_window.shown = False
            self.session.ui.processEvents()
        '''
        from chimerax.ui import tool_user_error
        structures = self.structures_list.value
        if not structures:
            self.tool_window.shown = True
            return tool_user_error("No structures chosen")
        from chimerax.atomic import AtomicStructure
        from chimerax.core.commands import run, concise_model_spec
        cmd = "kvfinder %s" % concise_model_spec(self.session, structures, relevant_types=AtomicStructure)
        from chimerax.core.commands import camel_case
        global _launch_settings
        for attr_name, default_value in self.cmd_defaults.items():
            cur_val = getattr(_launch_settings, attr_name)
            if attr_name == "grid_spacing" and (cur_val <= 0.0 or cur_val >= 5.0):
                self.tool_window.shown = True
                return tool_user_error("Grid spacing value must be > 0.0 and < 5.0")
            if cur_val != default_value:
                cmd += " " + camel_case(attr_name) + " %g" % cur_val
        if self.restrict_box.isChecked():
            cmd += " boxAtoms sel"
            padding = self.padding_box.value()
            if padding != self.cmd_defaults["box_pad"]:
                cmd += " boxPad %g" % padding
        if self.include_box.isChecked():
            cmd += " includeAtoms sel"
        if not self.replace_prev.isChecked():
            cmd += " replace false"
        run(self.session, cmd)
        '''
        if not apply:
            self.delete()

    def _chains_changed(self):
        changed_chains = {}
        for chain in self.chain_list.value:
            changed_chains.setdefault(chain.structure, []).append(chain)
        for s, chains in changed_chains.items():
            if len(chains) > 1:
                chains.remove(self._prev_chains[s])
                self.chain_list.blockSignals(True)
                self.chain_list.value = sum(changed_chains.values(), start=[])
                self.chain_list.blockSignals(False)
                break
        self._prev_chains = { chain.structure: chain for chain in self.chain_list.value }

