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
        from Qt.QtWidgets import QDoubleSpinBox, QRadioButton, QButtonGroup
        from Qt.QtCore import Qt
        self.layout = layout = QVBoxLayout()
        parent.setLayout(layout)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(2)

        layout.addWidget(QLabel("Chains to align"), alignment=Qt.AlignCenter)
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

        from chimerax.ui.options import SettingsPanel, FloatOption, SymbolicEnumOption, BooleanOption
        from chimerax.ui.options import Option, make_int_spinbox
        self.options_panel = panel = SettingsPanel(sorting=False, scrolled=False)
        tool_tips = {
            'cutoff_distance':
                "Residues whose principal atom are further apart\n"
                "than this distance will not be aligned in the\n"
                "generated sequence alignment",
            'column_criterion':
                "Whether a residue needs to match the distance cutoff to all other\n"
                "residues in its column, or just to one residue in the column",
            'gap_char':
                "Character used to depict gaps in the generated alignment",
        }
        class IterOption(BooleanOption):
            def get_value(self):
                return getattr(self.settings, self.attr_name)

            def set_value(self, value):
                if value == 0:
                    super().set_value(False)
                else:
                    super().set_value(True)

            value = property(get_value, set_value)

        # some of the min/max values are there to make the entry areas less wide
        for label, opt_type, attr_name, callback, kw in [
                ("Residue-residue distance cutoff (angstroms):", FloatOption, 'cutoff_distance', None,
                    {'min': 0.0, 'max': 99.9}),
                ("Residue aligned in column if within cutoff of:", SymbolicEnumOption, 'column_criterion',
                    None, { 'values': ("any", "all"), 'labels': ("at least one other", "all others") }),
                ("Gap character:", SymbolicEnumOption, 'gap_char', None,
                    { 'values': (".", "-", "~"), 'labels': (". (period)", "- (dash)", "~ (tilde)") }),
                ("Allow for circular permutation", BooleanOption, 'circular', None, {}),
                ("Iterate superposition/alignment...", IterOption, 'max_iterations', self._iterate_cb, {})]:
            opt = opt_type(label, getattr(_ma_settings, attr_name), callback,
                balloon=tool_tips.get(attr_name, None), attr_name=attr_name, settings=_ma_settings, **kw)
            setattr(self, attr_name + '_option', opt)
            panel.add_option(opt)
        self.circular_option.enabled = False
        gw, sub_panel = panel.add_option_group(group_label= "Iteration Parameters",
            group_alignment=Qt.AlignCenter)
        gw.setHidden(self.max_iterations_option.value == 0)
        gw_layout = QVBoxLayout()
        gw_layout.setContentsMargins(2,2,2,2)
        gw_layout.setSpacing(2)
        gw.setLayout(gw_layout)
        gw_layout.addWidget(sub_panel)
        self.iter_param_group = gw
        class IterNumOption(Option):
            def set_multiple(self):
                pass

            def _make_widget(self, *, main_option=None):
                self._main_option = main_option
                self.widget = layout = QVBoxLayout()
                layout.setSpacing(2)
                layout.setContentsMargins(0,0,0,0)
                self.num_iters_button_group = bg = QButtonGroup()
                row_layout = QHBoxLayout()
                row_layout.setSpacing(2)
                self.finite_iter_button = QRadioButton("at most")
                bg.addButton(self.finite_iter_button)
                row_layout.addWidget(self.finite_iter_button, alignment=Qt.AlignLeft)
                self.num_iters = make_int_spinbox(1, 999)
                self.num_iters.setValue(3)
                row_layout.addWidget(self.num_iters, alignment=Qt.AlignLeft)
                row_layout.addWidget(QLabel("times"), alignment=Qt.AlignLeft)
                row_layout.addStretch(1)
                layout.addLayout(row_layout)
                row_layout = QHBoxLayout()
                self.infinite_iter_button = QRadioButton("until convergence")
                bg.addButton(self.infinite_iter_button)
                row_layout.addWidget(self.infinite_iter_button, alignment=Qt.AlignLeft)
                row_layout.addStretch(1)
                layout.addLayout(row_layout)

            def get_value(self):
                if not self.main_option.isChecked():
                    return 0

                if self.finite_iter_button.isChecked():
                    return self.num_iters.value()
                return None

            def set_value(self, value):
                if value is None:
                    self.infinite_iter_button.setChecked(True)
                elif value != 0:
                    self.finite_iter_button.setChecked(True)
                    self.num_iters.setValue(value)

            value = property(get_value, set_value)

        sub_panel.add_option(IterNumOption("Iterate alignment:", _ma_settings.max_iterations,
            None, attr_name="max_iterations", settings=_ma_settings, main_option=self.max_iterations_option))
        layout.addWidget(panel, alignment=Qt.AlignCenter)

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

    def _iterate_cb(self, opt):
        self.iter_param_group.setHidden(self.max_iterations_option.value == 0)
