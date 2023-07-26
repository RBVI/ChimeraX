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

class Sequences(ToolInstance):

    #help = "help:user/tools/modelpanel.html"

    def __init__(self, session, tool_name):
        ToolInstance.__init__(self, session, tool_name)
        self.display_name = "Show Chain Sequence"
        self.settings = SequencesSettings(session, "ChainSequences")

        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area

        from Qt.QtWidgets import QVBoxLayout, QCheckBox
        layout = QVBoxLayout()
        parent.setLayout(layout)
        from chimerax.atomic.widgets import ChainListWidget
        self.chain_list = ChainListWidget(session, group_identical=self.settings.grouping,
            autoselect="first")
        self.chain_list.value_changed.connect(self._update_show_button)
        layout.addWidget(self.chain_list, stretch=1)

        self.grouping_button = QCheckBox("Group identical sequences")
        self.grouping_button.setChecked(self.settings.grouping)
        self.grouping_button.stateChanged.connect(self._grouping_change)
        layout.addWidget(self.grouping_button)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        bbox = qbbox()
        self._show_button = bbox.addButton("Show", qbbox.AcceptRole)
        bbox.addButton(qbbox.Cancel)
        #bbox.addButton(qbbox.Help)
        bbox.accepted.connect(self.show_seqs)
        bbox.accepted.connect(self.delete) # slots executed in the order they are connected
        bbox.rejected.connect(self.delete)
        from chimerax.core.commands import run
        bbox.helpRequested.connect(lambda *, run=run, ses=session: run(ses, "help " + self.help))
        layout.addWidget(bbox)

        self._update_show_button()

        tw.manage(placement=None)

    def show_seqs(self):
        groups = self.chain_list.grouped_value
        if groups:
            from chimerax.core.commands import run
            for chains in groups:
                run(self.session, "seq chain %s" % " ".join([chain.atomspec for chain in chains]))

    def _grouping_change(self, grouping):
        self.settings.grouping = grouping
        self.chain_list.group_identical = grouping

    def _update_show_button(self):
        self._show_button.setEnabled(bool(self.chain_list.value))

from chimerax.core.settings import Settings
class SequencesSettings(Settings):
    AUTO_SAVE = {
        'grouping': True
    }

_seqs = None
def sequences(session, tool_name):
    global _seqs
    if _seqs is None:
        _seqs = Sequences(session, tool_name)
    return _seqs
