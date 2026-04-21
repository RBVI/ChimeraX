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
from .settings import get_settings, defaults

class MatchAlignTool(ToolInstance):

    #help = "help:user/tools/findcavities.html"
    SESSION_SAVE = False

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)
        _ma_settings = get_settings(session)
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
        # some of the min/max values are there to make the entry areas less wide
        for label, opt_type, attr_name, callback, kw in [
                ("Residue-residue distance cutoff (angstroms):", FloatOption, 'cutoff_distance', None,
                    {'min': 0.0, 'max': 99.9}),
                ("Residue aligned in column if within cutoff of:", SymbolicEnumOption, 'column_criterion',
                    None, { 'values': ("any", "all"), 'labels': ("at least one other", "all others") }),
                ("Gap character:", SymbolicEnumOption, 'gap_char', None,
                    { 'values': (".", "-", "~"), 'labels': (". (period)", "- (dash)", "~ (tilde)") }),
                ("Allow for circular permutation", BooleanOption, 'circular', None, {}),
                ("Iterate superposition/alignment...", BooleanOption, 'iterate', self._iterate_cb, {})]:
            opt = opt_type(label, getattr(_ma_settings, attr_name), callback,
                balloon=tool_tips.get(attr_name, None), attr_name=attr_name, settings=_ma_settings, **kw)
            setattr(self, attr_name + '_option', opt)
            panel.add_option(opt)
        self.circular_option.enabled = False
        gw, sub_panel = panel.add_option_group(group_label= "Iteration Parameters",
            group_alignment=Qt.AlignCenter)
        gw.setHidden(not self.iterate_option.value)
        gw_layout = QVBoxLayout()
        gw_layout.setContentsMargins(2,2,2,2)
        gw_layout.setSpacing(2)
        gw.setLayout(gw_layout)
        gw_layout.addWidget(sub_panel)
        self.iter_param_group = gw
        class IterNumOption(Option):
            def set_multiple(self):
                pass

            def _make_widget(self):
                self.widget = layout = QVBoxLayout()
                layout.setSpacing(1)
                layout.setContentsMargins(2,2,2,2)
                self.num_iters_button_group = bg = QButtonGroup()
                row_layout = QHBoxLayout()
                row_layout.setSpacing(2)
                self.finite_iter_button = QRadioButton("at most")
                bg.addButton(self.finite_iter_button)
                row_layout.addWidget(self.finite_iter_button, alignment=Qt.AlignLeft)
                self.num_iters = make_int_spinbox(1, 999)
                num_iters = self.settings.max_iterations
                self.num_iters.setValue(defaults['max_iterations'] if num_iters is None else num_iters)
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
                if self.finite_iter_button.isChecked():
                    return self.num_iters.value()
                return None

            def set_value(self, value):
                if value is None:
                    self.infinite_iter_button.setChecked(True)
                else:
                    self.finite_iter_button.setChecked(True)
                    self.num_iters.setValue(value)

            value = property(get_value, set_value)

        self.max_iterations_option = IterNumOption("Iterate alignment:", _ma_settings.max_iterations, None,
            attr_name="max_iterations", settings=_ma_settings)
        sub_panel.add_option(self.max_iterations_option)

        class SuperimposeOption(Option):
            def set_multiple(self):
                pass

            def _make_widget(self):
                self.widget = layout = QVBoxLayout()
                layout.setSpacing(1)
                layout.setContentsMargins(2,2,2,2)
                self.superimpose_button_group = bg = QButtonGroup()
                row_layout = QHBoxLayout()
                self.entire_button = QRadioButton("across entire alignment")
                bg.addButton(self.entire_button)
                row_layout.addWidget(self.entire_button, alignment=Qt.AlignLeft)
                row_layout.addStretch(1)
                layout.addLayout(row_layout)
                row_layout = QHBoxLayout()
                row_layout.setSpacing(2)
                self.limited_button = QRadioButton("in stretches of at least")
                bg.addButton(self.limited_button)
                row_layout.addWidget(self.limited_button, alignment=Qt.AlignLeft)
                self.stretch_len = make_int_spinbox(1, 999)
                stretch_len = self.settings.min_stretch
                self.stretch_len.setValue(defaults['min_stretch'] if stretch_len is None else stretch_len)
                row_layout.addWidget(self.stretch_len, alignment=Qt.AlignLeft)
                row_layout.addWidget(QLabel("consecutive columns"), alignment=Qt.AlignLeft)
                row_layout.addStretch(1)
                layout.addLayout(row_layout)

            def get_value(self):
                if self.limited_button.isChecked():
                    return self.stretch_len.value()
                return None

            def set_value(self, value):
                if value is None:
                    self.entire_button.setChecked(True)
                else:
                    self.limited_button.setChecked(True)
                    self.stretch_len.setValue(value)

            value = property(get_value, set_value)

        self.min_stretch_option = SuperimposeOption("Superimpose full columns:", _ma_settings.min_stretch,
            None, attr_name="min_stretch", settings=_ma_settings)
        sub_panel.add_option(self.min_stretch_option)

        from chimerax.atomic.widgets import ChainMenuButton
        ref_chain_layout = QHBoxLayout()
        ref_chain_layout.addStretch(1)
        ref_chain_layout.addWidget(QLabel("Reference chain for matching:"))
        self.ref_chain_button = ChainMenuButton(session, no_value_button_text="",
            list_func=lambda cl=self.chain_list: cl.value,
            autoselect=ChainMenuButton.AUTOSELECT_FIRST)
        ref_chain_layout.addWidget(self.ref_chain_button)
        ref_chain_layout.addStretch(1)
        gw_layout.addLayout(ref_chain_layout)

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
        from chimerax.ui import tool_user_error
        chains = self.chain_list.value
        if len(chains) < 2:
            self.tool_window.shown = True
            return tool_user_error("Must choose at least two chains for alignment")
        from chimerax.atomic import concise_chain_spec
        args = [concise_chain_spec(chains)]

        def str_for_val(val):
            if isinstance(val, bool):
                return str(val).lower()
            if isinstance(val, (int, float)):
                return "%g" % val
            if val is None:
                return "none"
            from chimerax.core.commands import StringArg
            return StringArg.unparse(val)

        import inspect
        from .cmd import make_alignment
        cmd_params = inspect.signature(make_alignment).parameters
        from chimerax.core.commands import camel_case
        for kw_name in defaults.keys():
            if kw_name == 'iterate':
                continue
            default = cmd_params[kw_name].default
            val = getattr(self, kw_name + '_option').value
            if kw_name in ('max_iterations', 'min_stretch'):
                if not self.iterate_option.value:
                    if kw_name == 'max_iterations':
                        val = 0
                    else:
                        continue
            if kw_name == 'min_stretch' and val is None:
                val = 1
            if val != default:
                args.extend((camel_case(kw_name), str_for_val(val)))
        if self.iterate_option.value:
            ref_chain = self.ref_chain_button.value
            if not ref_chain:
                self.tool_window.shown = True
                return tool_user_error("Must choose a refence chain for iterative superposition")
            args.extend(('refChain', ref_chain.atomspec))

        from chimerax.core.commands import run
        run(self.session, "msa3d " + ' '.join(args))

        if not apply:
            self.delete()

    def _chains_changed(self):
        changed_chains = {}
        for chain in self.chain_list.value:
            changed_chains.setdefault(chain.structure, []).append(chain)
        for s, chains in changed_chains.items():
            if len(chains) > 1:
                if s in self._prev_chains:
                    chains.remove(self._prev_chains[s])
                self.chain_list.blockSignals(True)
                self.chain_list.value = sum(changed_chains.values(), start=[])
                self.chain_list.blockSignals(False)
                break
        self._prev_chains = { chain.structure: chain for chain in self.chain_list.value }
        self.ref_chain_button.refresh()

    def _iterate_cb(self, opt):
        self.iter_param_group.setHidden(not self.iterate_option.value)
