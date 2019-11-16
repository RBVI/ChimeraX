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
        self.ref_structure_list = AtomicStructureListWidget(session)
        self.ref_stacking.addWidget(self.ref_structure_list)
        self.match_structure_list = AtomicStructureListWidget(session)
        self.match_stacking.addWidget(self.match_structure_list)
        self.ref_chain_list = ChainListWidget(session, selection_mode='single')
        self.ref_stacking.addWidget(self.ref_chain_list)
        self.ref_chains_list = ChainListWidget(session, autoselect='single')
        self.ref_stacking.addWidget(self.ref_chains_list)
        self.matching_widgets = {
            CP_BEST_BEST: (self.ref_structure_list,  self.match_structure_list),
            CP_SPECIFIC_BEST: (self.ref_chain_list, self.match_structure_list),
            #TODO
            CP_SPECIFIC_SPECIFIC: (self.ref_chains_list, None),
        }

        from chimerax.ui.options import CategorizedSettingsPanel
        self.options = CategorizedSettingsPanel(category_sorting=False, option_sorting=False,
            category_scrolled={"Chain pairing": False})
        overall_layout.addWidget(self.options)
        from .settings import get_settings
        settings = get_settings(session)
        cp_opt = ChainPairingOption("", None, self._pairing_change, attr_name="chain_pairing",
            settings=settings, as_radio_buttons=True)
        self.options.add_option("Chain pairing", cp_opt)

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
