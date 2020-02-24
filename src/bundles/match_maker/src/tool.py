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

from .match import CP_SPECIFIC_SPECIFIC, CP_SPECIFIC_BEST, CP_BEST_BEST

class MatchMakerTool(ToolInstance):

    #help = "help:user/tools/hbonds.html"

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)
        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area
        from PyQt5.QtWidgets import QVBoxLayout, QGridLayout, QLabel, QDialogButtonBox, QStackedWidget
        from PyQt5.QtWidgets import QCheckBox
        from PyQt5.QtCore import Qt
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

        cp_opt = ChainPairingOption("", None, self._pairing_change, attr_name="chain_pairing",
            settings=settings, as_radio_buttons=True)
        self.options.add_option("Chain pairing", cp_opt)

        from chimerax.seqalign.align_algs.options import SeqAlignmentAlgOption
        self.options.add_option("Method", SeqAlignmentAlgOption("Sequence alignment algorithm", None, None,
            attr_name="alignment_algorithm", settings=settings))
        from chimerax.seqalign.sim_matrices.options import SimilarityMatrixNameOption
        self.options.add_option("Method", SimilarityMatrixNameOption("Matrix", None, None,
            attr_name="matrix", settings=settings))
        self.gap_open_option = IntOption("Gap opening penalty", None, None,
            attr_name="gap_open", settings=settings)
        self.options.add_option("Method", self.gap_open_option)
        self.options.add_option("Method", IntOption("Gap extension penalty", None, None,
            attr_name="gap_extend", settings=settings))
        ss_opt = BooleanOption("Include secondary structure score", None, self._include_ss_change,
            attr_name="use_ss", settings=settings)
        self.options.add_option("Method", ss_opt)
        self.compute_ss_option = BooleanOption("Compute secondary structure assignments", None, None,
            attr_name="compute_ss", settings=settings)
        self.options.add_option("Method", self.compute_ss_option)
        self.ss_ratio_option = FloatOption("Sequence vs. structure score weighting", None, None,
            attr_name="ss_mixture", settings=settings, as_slider=True, left_text="Residue similarity",
            right_text="Secondary structure", min=0.0, max=1.0, decimal_places=2)
        self.options.add_option("Method", self.ss_ratio_option)
        self._include_ss_change(ss_opt)

        from PyQt5.QtWidgets import QDialogButtonBox as qbbox
        bbox = qbbox(qbbox.Ok | qbbox.Apply | qbbox.Close | qbbox.Help)
        bbox.accepted.connect(self.run_matchmaker)
        bbox.button(qbbox.Apply).clicked.connect(self.run_matchmaker)
        bbox.accepted.connect(self.delete) # slots executed in the order they are connected
        bbox.rejected.connect(self.delete)
        from chimerax.core.commands import run
        #bbox.helpRequested.connect(lambda run=run, ses=session: run(ses, "help " + self.help))
        bbox.button(qbbox.Help).setEnabled(False)
        overall_layout.addWidget(bbox)

        self._pairing_change(cp_opt)
        tw.manage(placement=None)

    def run_matchmaker(self):
        from chimerax.core.commands import run
        #run(self.session, " ".join(self.gui.get_command()))

    def _include_ss_change(self, opt):
        self.gap_open_option.enabled = not opt.value
        self.compute_ss_option.enabled = opt.value
        self.ss_ratio_option.enabled = opt.value
        if opt.value:
            from .settings import get_settings
            settings = get_settings(self.session)
            self.compute_ss_option.value = settings.compute_ss
        else:
            self.compute_ss_option.value = False

    def _pairing_change(self, opt):
        ref_label, match_label = self.label_texts[opt.value]
        self.ref_label.setText(ref_label)
        self.match_label.setText(match_label)
        ref_widget, match_widget = self.matching_widgets[opt.value]
        self.ref_stacking.setCurrentWidget(ref_widget)
        if match_widget is not None:
            self.match_stacking.setCurrentWidget(match_widget)

from chimerax.ui.options import SymbolicEnumOption
class ChainPairingOption(SymbolicEnumOption):
    labels = (
        "Best-aligning pair of chains between reference and match structure",
        "Specific chain in reference structure and best-aligning chain in match structure",
        "Specific chain(s) in reference structure with specific chain(s) in match structure"
    )
    values = (CP_BEST_BEST, CP_SPECIFIC_BEST, CP_SPECIFIC_SPECIFIC)

from PyQt5.QtWidgets import QWidget
class ChainListsWidget(QWidget):
    def __init__(self, session, *args, **kw):
        super().__init__(*args, **kw)
        self.__session = session
        from PyQt5.QtWidgets import QVBoxLayout, QLabel
        self.__work_layout = QVBoxLayout()
        self.__work_layout.setSpacing(0)
        self.__container_layout = QVBoxLayout() # so that I can replace the working layout
        self.__container_layout.addLayout(self.__work_layout)
        self.setLayout(self.__container_layout)
        self.__chain_list_mapping = {}
        self.__empty_label = QLabel("Choose one or more reference chains in the list on the left.  To"
            " specify multiple pairs, choose multiple reference chains at once; a menu for choosing the"
            " corresponding match chain will be provided for each reference chain.")
        self.__empty_label.setWordWrap(True)
        self.__work_layout.addWidget(self.__empty_label)

    def update(self, ref_chains):
        if not self.__chain_list_mapping:
            self.__empty_label.hide()
        if not ref_chains:
            for widgets in self.__chain_list_mapping.values():
                for widget in widgets:
                    self.__work_layout.removeWidget(widget)
                    widget.deleteLater()
            self.__empty_label.show()
            return
        # rearranging widgets in an existing layout is nigh impossible so...
        from PyQt5.QtWidgets import QVBoxLayout, QLabel
        next_layout = QVBoxLayout()
        next_layout.setSpacing(0)
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
            from PyQt5.QtCore import Qt
            if next_mapping:
                next_layout.addSpacing(10)
            next_layout.addWidget(label, alignment=Qt.AlignBottom)
            next_layout.addWidget(chain_list, alignment=Qt.AlignTop)
            next_mapping[chain] = (label, chain_list)
        for widgets in self.__chain_list_mapping.values():
            for widget in widgets:
                widget.deleteLater()
        self.__chain_list_mapping = next_mapping
        self.__container_layout.takeAt(0)
        self.__container_layout.addLayout(next_layout)
        self.__work_layout = next_layout
