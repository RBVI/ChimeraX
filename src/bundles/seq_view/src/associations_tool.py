# vim: set expandtab ts=4 sw=4:

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

class AssociationsTool:

    def __init__(self, sv, tool_window):
        self.sv = sv
        self.tool_window = tool_window
        tool_window.help = "help:user/tools/sequenceviewer.html#association"

        from PyQt5.QtWidgets import QHBoxLayout
        layout = QHBoxLayout()
        from chimerax.core.atomic.widgets import ChainListWidget
        self.chain_list = ChainListWidget(sv.session, selection_mode='single')
        self.chain_list.value_changed.connect(self._chain_changed)
        layout.addWidget(self.chain_list)

        from chimerax.seqalign.widgets import AlignSeqMenuButton
        self.best_assoc_label = "Best matching sequence"
        self.assoc_button = AlignSeqMenuButton(sv.alignment, no_value_button_text="Not associated",
            no_value_menu_text="(none)", special_items=[self.best_assoc_label])
        self.assoc_button.value_changed.connect(self._seq_changed)
        layout.addWidget(self.assoc_button)

        tool_window.ui_area.setLayout(layout)

        # get initial assoc info correct
        self.assoc_button.blockSignals(True)
        self.chain_list.blockSignals(True)
        self._chain_changed()
        self.chain_list.blockSignals(False)
        self.assoc_button.blockSignals(False)

    def _assoc_mod(self, note_data):
        # called from sequence viewer if associations modified
        self._chain_changed()

    def _chain_changed(self):
        if self.chain_list.count() == 0:
            self.tool_window.shown = False
        self.assoc_button.value = self.sv.alignment.associations.get(self.chain_list.value, None)

    def _seq_changed(self):
        # this can also get called if sequences get deleted, so try to do some checking
        chain = self.chain_list.value
        if not chain:
            return
        cur_assoc = self.sv.alignment.associations.get(chain, None)
        req_assoc = self.assoc_button.value
        if cur_assoc == req_assoc:
            return
        from chimerax.core.commands import run
        if not req_assoc:
            run(self.sv.session, "sequence disassoc %s" % chain.string(style="command"))
        elif req_assoc == self.best_assoc_label:
            run(self.sv.session, "sequence assoc %s %s" % (chain.string(style="command"),
                self.sv.alignment.ident))
        else:
            run(self.sv.session, "sequence assoc %s %s:%d" % (chain.string(style="command"),
                self.sv.alignment.ident, self.sv.alignment.seqs.index(req_assoc)+1))
