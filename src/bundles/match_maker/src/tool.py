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

from .match import CP_SPECIFIC_SPECIFIC, CP_SPECIFIC_BEST, CP_BEST_BEST
from Qt.QtCore import Qt, Signal

class MatchMakerTool(ToolInstance):

    help = "help:user/tools/matchmaker.html"

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)
        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area
        from Qt.QtWidgets import QVBoxLayout, QGridLayout, QLabel, QDialogButtonBox, QStackedWidget
        from Qt.QtWidgets import QCheckBox
        overall_layout = QVBoxLayout()
        overall_layout.setContentsMargins(0,0,0,0)
        overall_layout.setSpacing(0)
        parent.setLayout(overall_layout)

        matching_layout = QGridLayout()
        matching_layout.setContentsMargins(0,0,0,0)
        matching_layout.setSpacing(0)
        matching_layout.setColumnStretch(0, 1)
        matching_layout.setColumnStretch(1, 1)
        matching_layout.setRowStretch(1, 1)
        overall_layout.addLayout(matching_layout, stretch=1)
        self.label_texts = {
            CP_BEST_BEST: ("Reference structure:", "Structure(s) to match:"),
            CP_SPECIFIC_BEST: ("Reference chain:", "Structure(s) to match:"),
            CP_SPECIFIC_SPECIFIC: ("Reference chain(s):", "Chain(s) to match:")
        }
        self.ref_label = QLabel()
        matching_layout.addWidget(self.ref_label, 0, 0, alignment=Qt.AlignLeft)
        self.match_label = QLabel()
        matching_layout.addWidget(self.match_label, 0, 1, alignment=Qt.AlignLeft)
        self.ref_stacking = QStackedWidget()
        matching_layout.addWidget(self.ref_stacking, 1, 0)
        self.match_stacking = QStackedWidget()
        matching_layout.addWidget(self.match_stacking, 1, 1)
        from chimerax.atomic.widgets import AtomicStructureListWidget, ChainListWidget
        self.ref_structure_list = AtomicStructureListWidget(session, selection_mode='single')
        self.ref_stacking.addWidget(self.ref_structure_list)
        self.match_structure_structure_list = AtomicStructureListWidget(session,
            filter_func=lambda s, ref_list=self.ref_structure_list: s != ref_list.value)
        self.ref_structure_list.value_changed.connect(self.match_structure_structure_list.refresh)
        self.match_stacking.addWidget(self.match_structure_structure_list)
        self.ref_chain_list = ChainListWidget(session, selection_mode='single')
        self.ref_stacking.addWidget(self.ref_chain_list)
        self.match_chain_structure_list = AtomicStructureListWidget(session, filter_func=lambda s,
            ref_list=self.ref_chain_list: s != getattr(ref_list.value, 'structure', None))
        self.ref_chain_list.value_changed.connect(self.match_chain_structure_list.refresh)
        self.match_stacking.addWidget(self.match_chain_structure_list)
        self.ref_chains_list = ChainListWidget(session, autoselect='single')
        self.ref_stacking.addWidget(self.ref_chains_list)
        self.match_chain_lists = ChainListsWidget(session)
        self.ref_chains_list.value_changed.connect(
            lambda ref=self.ref_chains_list:self.match_chain_lists.update(ref.value))
        self.match_stacking.addWidget(self.match_chain_lists)
        self.matching_widgets = {
            CP_BEST_BEST: (self.ref_structure_list,  self.match_structure_structure_list),
            CP_SPECIFIC_BEST: (self.ref_chain_list, self.match_chain_structure_list),
            CP_SPECIFIC_SPECIFIC: (self.ref_chains_list, self.match_chain_lists),
        }
        self.ref_sel_restrict = QCheckBox("Also restrict to selection")
        matching_layout.addWidget(self.ref_sel_restrict, 2, 0, alignment=Qt.AlignCenter)
        self.match_sel_restrict = QCheckBox("Also restrict to selection")
        matching_layout.addWidget(self.match_sel_restrict, 2, 1, alignment=Qt.AlignCenter)

        from chimerax.ui.options import CategorizedSettingsPanel, IntOption, BooleanOption, FloatOption
        self.options = CategorizedSettingsPanel(category_sorting=False, option_sorting=False,
            category_scrolled={"Chain pairing": False})
        overall_layout.addWidget(self.options, stretch=1)
        from .settings import get_settings
        settings = get_settings(session)

        self.chain_pairing_option = cp_opt = ChainPairingOption("", None, self._pairing_change,
            attr_name="chain_pairing", settings=settings, as_radio_buttons=True)
        self.options.add_option("Chain pairing", cp_opt)

        from chimerax.alignment_algs.options import SeqAlignmentAlgOption
        self.options.add_option("Alignment", BooleanOption("Show pairwise sequence alignment(s)", None, None,
            attr_name="show_alignment", settings=settings))
        self.options.add_option("Alignment", SeqAlignmentAlgOption("Sequence alignment algorithm",
            None, None, attr_name="alignment_algorithm", settings=settings))
        from chimerax.sim_matrices.options import SimilarityMatrixNameOption
        self.options.add_option("Alignment", SimilarityMatrixNameOption("Matrix", None, None,
            attr_name="matrix", settings=settings))
        self.gap_open_option = IntOption("Gap opening penalty", None, None,
            attr_name="gap_open", settings=settings)
        self.options.add_option("Alignment", self.gap_open_option)
        self.options.add_option("Alignment", IntOption("Gap extension penalty", None, None,
            attr_name="gap_extend", settings=settings))
        ss_opt = BooleanOption("Include secondary structure score", None, self._include_ss_change,
            attr_name="use_ss", settings=settings)
        self.options.add_option("Alignment", ss_opt)
        self.ss_widget, ss_options = self.options.add_option_group("Alignment", sorting=False,
            group_label="Secondary structure scoring")
        ss_layout = QVBoxLayout()
        ss_layout.addWidget(ss_options)
        self.ss_widget.setLayout(ss_layout)
        self.compute_ss_option = BooleanOption("Compute secondary structure assignments", None,
            self._compute_ss_change, attr_name="compute_ss", settings=settings)
        ss_options.add_option(self.compute_ss_option)
        self.overwrite_ss_option = BooleanOption("Overwrite previous assignments", None, None,
            attr_name="overwrite_ss", settings=settings)
        ss_options.add_option(self.overwrite_ss_option)
        self.ss_ratio_option = FloatOption("Secondary structure weighting", None, None,
            attr_name="ss_mixture", settings=settings, as_slider=True, left_text="Residue similarity",
            right_text="Secondary structure", min=0.0, max=1.0, decimal_places=2, ignore_wheel_event=True)
        ss_options.add_option(self.ss_ratio_option)
        self.ss_matrix_option = SSScoringMatrixOption("Scoring matrix", None, None,
            attr_name='ss_scores', settings=settings)
        ss_options.add_option(self.ss_matrix_option)
        self.ss_helix_gap_option = IntOption("Intra-helix gap opening penalty", None, None,
            attr_name='helix_open', settings=settings)
        ss_options.add_option(self.ss_helix_gap_option)
        self.ss_strand_gap_option = IntOption("Intra-strand gap opening penalty", None, None,
            attr_name='strand_open', settings=settings)
        ss_options.add_option(self.ss_strand_gap_option)
        self.ss_other_gap_option = IntOption("Any other gap opening penalty", None, None,
            attr_name='other_open', settings=settings)
        ss_options.add_option(self.ss_other_gap_option)
        self._include_ss_change(ss_opt)

        iter_opt = BooleanOption("Iterate by pruning long atom pairs", None, self._iterate_change,
            attr_name="iterate", settings=settings)
        self.options.add_option("Fitting", iter_opt)
        self.iter_cutoff_option = FloatOption("Iteration cutoff distance", None, None,
            attr_name="iter_cutoff", settings=settings)
        self.options.add_option("Fitting", self.iter_cutoff_option)
        self._iterate_change(iter_opt)
        self.options.add_option("Fitting", BooleanOption("Verbose logging", None, None,
            attr_name="verbose_logging", settings=settings))
        bring_container, bring_options = self.options.add_option_group("Fitting",
            group_alignment=Qt.AlignHCenter|Qt.AlignTop)
        bring_layout = QVBoxLayout()
        bring_container.setLayout(bring_layout)
        self.bring_label = QLabel("If one model being matched, also move these models along with it:")
        bring_layout.addWidget(self.bring_label)
        from chimerax.ui.widgets import ModelListWidget
        self.bring_model_list = ModelListWidget(session, filter_func=self._filter_bring_models,
            autoselect=None)
        for ref_widget, match_widget in self.matching_widgets.values():
            ref_widget.value_changed.connect(self.bring_model_list.refresh)
            match_widget.value_changed.connect(self.bring_model_list.refresh)
            match_widget.value_changed.connect(self._match_value_change)
        bring_layout.addWidget(self.bring_model_list)
        self._match_value_change()

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        bbox = qbbox(qbbox.Ok | qbbox.Apply | qbbox.Close | qbbox.Help)
        bbox.accepted.connect(self.run_matchmaker)
        bbox.button(qbbox.Apply).clicked.connect(self.run_matchmaker)
        bbox.rejected.connect(self.delete)
        from chimerax.core.commands import run
        bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help " + self.help))
        overall_layout.addWidget(bbox)

        self._pairing_change(cp_opt)
        tw.manage(placement=None)

    def run_matchmaker(self):
        from chimerax.core.commands import StringArg, BoolArg, FloatArg, DynamicEnum, NoneArg
        from .settings import defaults, get_settings
        settings = get_settings(self.session)
        chain_pairing = self.chain_pairing_option.value
        ref_widget, match_widget = self.matching_widgets[chain_pairing]
        ref_value = ref_widget.value
        match_value = match_widget.value
        if chain_pairing == CP_SPECIFIC_SPECIFIC:
            ref_spec = "".join([rchain.atomspec for rchain, mchain in match_value])
        else:
            ref_spec = None if ref_value is None else ref_value.atomspec
        if not ref_spec or not match_value:
            raise UserError("No reference and/or match structure/chain chosen")
        if self.ref_sel_restrict.isChecked():
            ref_spec = ref_spec + " & sel"
        if chain_pairing == CP_SPECIFIC_SPECIFIC:
            match_spec = "".join([mchain.atomspec for rchain, mchain in match_value])
        else:
            from chimerax.core.commands import concise_model_spec
            match_spec = concise_model_spec(self.session, match_value)
        if not match_spec:
            raise UserError("No match structure/chain(s) chosen")
        if self.match_sel_restrict.isChecked():
            match_spec = match_spec + " & sel"

        cmd = "matchmaker " + match_spec + " to " + ref_spec
        if chain_pairing != defaults['chain_pairing']:
            cmd += ' pairing ' + chain_pairing

        alg = settings.alignment_algorithm
        if alg != defaults['alignment_algorithm']:
            cmd += ' alg ' + StringArg.unparse(alg)

        verbose = settings.verbose_logging
        if verbose != defaults['verbose_logging']:
            cmd += ' verbose ' + BoolArg.unparse(verbose)

        use_ss = settings.use_ss
        if use_ss:
            ss_fraction = settings.ss_mixture
            if ss_fraction != defaults['ss_mixture']:
                cmd += ' ssFraction ' + FloatArg.unparse(ss_fraction)
        else:
            cmd += ' ssFraction ' + BoolArg.unparse(use_ss)

        matrix = settings.matrix
        if matrix != defaults['matrix']:
            from chimerax import sim_matrices
            cmd += ' matrix ' + DynamicEnum(
                lambda ses=self.session: sim_matrices.matrices(ses.logger).keys()).unparse(matrix)

        gap_open = settings.gap_open
        if not use_ss and gap_open != defaults['gap_open']:
            cmd += ' gapOpen ' + FloatArg.unparse(gap_open)

        helix_open = settings.helix_open
        from chimerax.core.commands import run
        if helix_open != defaults['helix_open']:
            cmd += ' hgap ' + FloatArg.unparse(helix_open)

        strand_open = settings.strand_open
        from chimerax.core.commands import run
        if strand_open != defaults['strand_open']:
            cmd += ' sgap ' + FloatArg.unparse(strand_open)

        other_open = settings.other_open
        from chimerax.core.commands import run
        if other_open != defaults['other_open']:
            cmd += ' ogap ' + FloatArg.unparse(other_open)

        iterate = settings.iterate
        if iterate:
            iter_cutoff = settings.iter_cutoff
            if iter_cutoff != defaults['iter_cutoff']:
                cmd += ' cutoffDistance ' + FloatArg.unparse(iter_cutoff)
        else:
            cmd += ' cutoffDistance ' + NoneArg.unparse(None)

        gap_extend = settings.gap_extend
        if gap_extend != defaults['gap_extend']:
            cmd += ' gapExtend ' + FloatArg.unparse(gap_extend)

        if self.bring_model_list.isEnabled():
            models = self.bring_model_list.value
            if models:
                cmd += ' bring ' + concise_model_spec(self.session, models)

        show_alignment = settings.show_alignment
        if show_alignment != defaults['show_alignment']:
            cmd += ' showAlignment ' + BoolArg.unparse(show_alignment)

        compute_ss = settings.compute_ss
        if compute_ss != defaults['compute_ss']:
            cmd += ' computeSS ' + BoolArg.unparse(compute_ss)

        overwrite_ss = settings.overwrite_ss
        if compute_ss and overwrite_ss != defaults['overwrite_ss']:
            cmd += ' keepComputedSS ' + BoolArg.unparse(overwrite_ss)

        ss_matrix = settings.ss_scores
        if ss_matrix != defaults['ss_scores']:
            for key, val in ss_matrix.items():
                order = ['H', 'S', 'O']
                if val != defaults['ss_scores'][key]:
                    let1, let2 = key
                    if order.index(let1) > order.index(let2):
                        continue
                    cmd += ' mat' + let1 + let2 + ' ' + FloatArg.unparse(val)

        run(self.session, cmd)
        self.delete()

    def _compute_ss_change(self, opt):
        if opt.value:
            from .settings import get_settings
            settings = get_settings(self.session)
            self.overwrite_ss_option.value = settings.overwrite_ss
        else:
            self.overwrite_ss_option.value = False
        self.overwrite_ss_option.enabled = opt.value

    def _filter_bring_models(self, model):
        from chimerax.atomic import PseudobondGroup
        if isinstance(model, PseudobondGroup):
            return False
        chain_pairing = self.chain_pairing_option.value
        ref_widget, match_widget = self.matching_widgets[chain_pairing]
        ref_value = ref_widget.value
        match_value = match_widget.value
        if chain_pairing == CP_BEST_BEST:
            ref_structures = [ref_value]
            match_structures = match_value
        elif chain_pairing == CP_SPECIFIC_BEST:
            ref_structures = [ref_value.structure]
            match_structures = match_value
        else:
            ref_structures = set()
            match_structures = set()
            for rc, mc in match_value:
                ref_structures.add(rc.structure)
                match_structures.add(mc.structure)
        for mstructure in match_structures:
            # can't be in the same hierarchy as the match model
            if model in mstructure.all_models() or mstructure in model.all_models():
                return False
        for rstructure in ref_structures:
            # can't be above the reference structure in the hierarchy
            if rstructure in model.all_models():
                return False
        return True

    def _include_ss_change(self, opt):
        self.ss_widget.setHidden(not opt.value)
        self.gap_open_option.enabled = not opt.value
        if opt.value:
            from .settings import get_settings
            settings = get_settings(self.session)
            self.compute_ss_option.value = settings.compute_ss
        else:
            self.compute_ss_option.value = False
        self._compute_ss_change(self.compute_ss_option)

    def _iterate_change(self, opt):
        self.iter_cutoff_option.enabled = opt.value

    def _match_value_change(self):
        chain_pairing = self.chain_pairing_option.value
        ref_widget, match_widget = self.matching_widgets[chain_pairing]
        match_value = match_widget.value
        if chain_pairing == CP_BEST_BEST:
            match_structures = match_value
        elif chain_pairing == CP_SPECIFIC_BEST:
            match_structures = match_value
        else:
            match_structures = set()
            for rc, mc in match_value:
                match_structures.add(mc.structure)
        enable = len(match_structures) == 1
        self.bring_label.setEnabled(enable)
        self.bring_model_list.setEnabled(enable)

    def _pairing_change(self, opt):
        ref_label, match_label = self.label_texts[opt.value]
        self.ref_label.setText(ref_label)
        self.match_label.setText(match_label)
        ref_widget, match_widget = self.matching_widgets[opt.value]
        self.ref_stacking.setCurrentWidget(ref_widget)
        self.match_stacking.setCurrentWidget(match_widget)
        self.bring_model_list.refresh()

from chimerax.ui.options import SymbolicEnumOption, Option
class ChainPairingOption(SymbolicEnumOption):
    labels = (
        "Best-aligning pair of chains between reference and match structure",
        "Specific chain in reference structure and best-aligning chain in match structure",
        "Specific chain(s) in reference structure with specific chain(s) in match structure"
    )
    values = (CP_BEST_BEST, CP_SPECIFIC_BEST, CP_SPECIFIC_SPECIFIC)

class SSScoringMatrixOption(Option):

    def get_value(self):
        matrix = {}
        for key, cell in self._cells.items():
            if not cell.hasAcceptableInput():
                raise UserError("Secondary structure scoring matrix entry %s contains non-numeric value"
                    % key)
            val = float(cell.text())
            matrix[key] = val
            matrix[(key[1], key[0])] = val
        return matrix

    def set_value(self, matrix):
        for key, val in matrix.items():
            self._cells[key].setText("%g" % val)

    value = property(get_value, set_value)

    def set_multiple(self):
        pass

    def _make_widget(self, **kw):
        from Qt.QtWidgets import QFrame, QGridLayout, QLineEdit, QLabel
        from Qt.QtGui import QDoubleValidator
        from Qt.QtCore import Qt
        self.widget = QFrame()
        self._cells = {}
        cell_order = ['H', 'S', 'O']
        layout_data = [
            [''] + cell_order,
            ["Helix", None, "-", "-"],
            ["Strand", None, None, "-"],
            ["Other", None, None, None]
        ]
        grid = QGridLayout()
        grid.setContentsMargins(2,2,2,2)
        grid.setSpacing(1)
        validator = QDoubleValidator()
        for ri, row_data in enumerate(layout_data):
            for ci, cell_data in enumerate(row_data):
                if cell_data is None:
                    cell = QLineEdit()
                    cell.setValidator(validator)
                    cell.setFixedWidth(30)
                    cell.setAlignment(Qt.AlignCenter)
                    cell.textEdited.connect(
                        lambda *args, cell=cell, cb=self.make_callback: cell.hasAcceptableInput() and cb())
                    grid.addWidget(cell, ri, ci, alignment=Qt.AlignCenter)
                    row_letter = cell_order[ri-1]
                    col_letter = cell_order[ci-1]
                    self._cells[(row_letter, col_letter)] = cell
                    self._cells[(col_letter, row_letter)] = cell
                else:
                    text = QLabel(cell_data)
                    if ci == 0:
                        grid.addWidget(text, ri, ci, alignment=Qt.AlignRight)
                    else:
                        grid.addWidget(text, ri, ci, alignment=Qt.AlignCenter)
        self.widget.setLayout(grid)

from Qt.QtWidgets import QWidget
class ChainListsWidget(QWidget):

    value_changed = Signal()

    def __init__(self, session, *args, **kw):
        super().__init__(*args, **kw)
        self.__session = session
        from Qt.QtWidgets import QVBoxLayout, QLabel, QWidget, QScrollArea
        self.__work_layout = QVBoxLayout()
        self.__work_layout.setSpacing(0)
        self.__work_layout.setContentsMargins(0,0,0,0)
        self.__container_layout = QVBoxLayout() # so that I can replace the working layout
        self.__container_layout.setContentsMargins(0,0,0,0)
        self.__container_layout.addLayout(self.__work_layout)
        self.__scroll_layout = QVBoxLayout()
        self.__scroll_layout.setContentsMargins(0,0,0,0)
        self.setLayout(self.__scroll_layout)
        self.__scroll_area = QScrollArea()
        scrolled_widget = QWidget()
        self.__scroll_area.setWidget(scrolled_widget)
        self.__container_layout.setSizeConstraint(QVBoxLayout.SetMinAndMaxSize)
        scrolled_widget.setLayout(self.__container_layout)
        self.__chain_list_mapping = {}
        self.__empty_label = QLabel("Choose one or more reference chains in the list on the left.  To"
            " specify multiple pairs, choose multiple reference chains at once; a menu for choosing the"
            " corresponding match chain will be provided for each reference chain.")
        self.__empty_label.setWordWrap(True)
        self.__shown_widget = None
        self.__show_widget(self.__empty_label)

    def update(self, ref_chains):
        if not ref_chains:
            for widgets in self.__chain_list_mapping.values():
                for widget in widgets:
                    self.__work_layout.removeWidget(widget)
                    widget.destroy()
            self.__chain_list_mapping.clear()
            self.__show_widget(self.__empty_label)
            self.value_changed.emit()
            return
        # rearranging widgets in an existing layout is nigh impossible so...
        self.__show_widget(self.__scroll_area)
        from Qt.QtWidgets import QVBoxLayout, QLabel
        next_layout = QVBoxLayout()
        next_layout.setSpacing(0)
        next_layout.setContentsMargins(0,0,0,0)
        self.__work_layout.removeWidget(self.__empty_label)
        next_layout.addWidget(self.__empty_label)
        next_mapping = {}
        for chain in ref_chains:
            if chain in self.__chain_list_mapping:
                label, chain_list = self.__chain_list_mapping.pop(chain)
            else:
                label = QLabel("ref: %s" % chain)
                from chimerax.atomic.widgets import ChainMenuButton
                chain_list = ChainMenuButton(self.__session,
                    filter_func=lambda c, ref=chain: c.structure != ref.structure)
                chain_list.value_changed.connect(self.value_changed.emit)
            from Qt.QtCore import Qt
            if next_mapping:
                next_layout.addSpacing(10)
            next_layout.addWidget(label, alignment=Qt.AlignBottom)
            next_layout.addWidget(chain_list, alignment=Qt.AlignTop)
            next_mapping[chain] = (label, chain_list)
        for widgets in self.__chain_list_mapping.values():
            for widget in widgets:
                widget.hide()
                widget.destroy()
        self.__chain_list_mapping = next_mapping
        self.__container_layout.takeAt(0)
        self.__container_layout.addLayout(next_layout)
        self.__work_layout = next_layout
        self.value_changed.emit()

    @property
    def value(self):
        val = []
        for ref_chain, match_info in self.__chain_list_mapping.items():
            match_label, match_menu = match_info
            match_chain = match_menu.value
            if match_chain is not None:
                val.append((ref_chain, match_chain))
        return val

    def __show_widget(self, widget):
        if widget == self.__shown_widget:
            return
        if self.__shown_widget is not None:
            self.__scroll_layout.removeWidget(self.__shown_widget)
            self.__shown_widget.hide()
        self.__scroll_layout.addWidget(widget)
        widget.show()
        self.__shown_widget = widget
