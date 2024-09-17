# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

class AssociationsTool:

    multiseq_text = "(various)"

    def __init__(self, sv, tool_window):
        self.sv = sv
        self.tool_window = tool_window
        tool_window.help = "help:user/tools/sequenceviewer.html#association"

        from Qt.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QWidget
        from Qt.QtCore import Qt
        layout = QHBoxLayout()

        # Widgets for multi-sequence alignment
        self.multi_seq_area = QWidget()
        layout.addWidget(self.multi_seq_area)
        ms_layout = QHBoxLayout()
        ms_layout.setSpacing(2)
        self.multi_seq_area.setLayout(ms_layout)

        from chimerax.atomic.widgets import ChainListWidget
        self.chain_list = ChainListWidget(sv.session, autoselect=ChainListWidget.AUTOSELECT_SINGLE)
        self.chain_list.value_changed.connect(self._chain_changed)
        ms_layout.addWidget(self.chain_list)

        menu_layout = QVBoxLayout()
        ms_layout.addLayout(menu_layout)

        menu_layout.addStretch(1)

        self.pick_a_chain = QLabel("Choose one or more\nchains from the left")
        menu_layout.addWidget(self.pick_a_chain)

        self.assoc_button_header = QLabel("Associated sequence")
        menu_layout.addWidget(self.assoc_button_header, alignment=Qt.AlignBottom|Qt.AlignHCenter)
        from chimerax.seqalign.widgets import AlignSeqMenuButton
        not_associated_text = "Not associated"
        class AllowMultiASMB(AlignSeqMenuButton):
            def showEvent(s, *args):
                if s.text() == self.multiseq_text:
                    s.setText(not_associated_text)
                super().showEvent(*args)
                self.assoc_button.blockSignals(True)
                self.chain_list.blockSignals(True)
                self._chain_changed()
                self.chain_list.blockSignals(False)
                self.assoc_button.blockSignals(False)

        self.best_assoc_label = "Best-matching sequence"
        self.assoc_button = AllowMultiASMB(sv.alignment, no_value_button_text=not_associated_text,
            no_value_menu_text="(none)", special_items=[self.best_assoc_label])
        self.assoc_button.value_changed.connect(self._seq_changed)
        menu_layout.addWidget(self.assoc_button, alignment=Qt.AlignTop|Qt.AlignHCenter)

        menu_layout.addStretch(3)

        # Widgets for single sequence
        self.single_seq_area = QWidget()
        layout.addWidget(self.single_seq_area)
        ss_layout = QVBoxLayout()
        self.single_seq_area.setLayout(ss_layout)

        ss_layout.addWidget(
            QLabel("Chains chosen/unchosen below will be associated/dissociated immediately"))

        from chimerax.atomic.widgets import ChainListWidget
        self.ss_chain_list = ChainListWidget(sv.session, autoselect=ChainListWidget.AUTOSELECT_NONE)
        self.ss_chain_list.value_changed.connect(self._ss_chain_changed)
        ss_layout.addWidget(self.ss_chain_list)
        self._processing_ss_list = False

        self._choose_widgets()

        tool_window.ui_area.setLayout(layout)

        # get initial assoc info correct
        self.assoc_button.blockSignals(True)
        self.chain_list.blockSignals(True)
        self._chain_changed()
        self.chain_list.blockSignals(False)
        self.assoc_button.blockSignals(False)

        self._set_ss_data()

    def _align_arg(self):
        if len(self.sv.session.alignments) > 1:
            return ' ' + self.sv.alignment.ident
        return ''

    def _assoc_mod(self, note_data):
        # called from sequence viewer if associations modified
        self._chain_changed()
        if not self._processing_ss_list:
            self._set_ss_data()

    def _chain_changed(self):
        self.assoc_button.blockSignals(True)
        if self.chain_list.count() == 0:
            self.tool_window.shown = False
        if self.assoc_button.text() == self.multiseq_text:
            # need to do this to avoid button try to get the "chain" associated with the text
            # (for equality testing)
            self.assoc_button.value = None
        chains = self.chain_list.value
        if len(chains) == 0:
            self.assoc_button.value = None
            show_button = False
        elif len(chains) == 1:
            self.assoc_button.value = self.sv.alignment.associations.get(chains[0], None)
            show_button = True
        else:
            values = set([self.sv.alignment.associations.get(chain, None) for chain in chains])
            if len(values) == 1:
                self.assoc_button.value = values.pop()
            else:
                self.assoc_button.setText(self.multiseq_text)
            show_button = True
        self.assoc_button_header.setHidden(not show_button)
        self.assoc_button.setHidden(not show_button)
        self.pick_a_chain.setHidden(show_button)
        self.assoc_button.blockSignals(False)

    def _choose_widgets(self):
        # also called from sequence viewer if sequences added/deleted
        show_single = len(self.sv.alignment.seqs) == 1
        self.multi_seq_area.setHidden(show_single)
        self.single_seq_area.setHidden(not show_single)

    def _seq_changed(self):
        # this can also get called if sequences get deleted, so try to do some checking
        chains = self.chain_list.value
        if not chains:
            return
        if self.assoc_button.text() == self.multiseq_text:
            from chimerax.core.errors import UserError
            raise UserError("Choose a specific sequence to associate with")
        req_assoc = self.assoc_button.value
        self.assoc_button.blockSignals(True)
        self.chain_list.blockSignals(True)
        for chain in chains:
            cur_assoc = self.sv.alignment.associations.get(chain, None)
            if cur_assoc == req_assoc:
                continue
            from chimerax.core.commands import run
            if not req_assoc:
                run(self.sv.session, "sequence disassoc %s%s" % (chain.string(style="command"),
                    self._align_arg()))
            elif req_assoc == self.best_assoc_label:
                run(self.sv.session, "sequence assoc %s%s" % (chain.string(style="command"),
                    self._align_arg()))
            else:
                run(self.sv.session, "sequence assoc %s %s:%d" % (chain.string(style="command"),
                    self.sv.alignment.ident, self.sv.alignment.seqs.index(req_assoc)+1))
        self.chain_list.blockSignals(False)
        self.assoc_button.blockSignals(False)
        self._chain_changed()

    def _set_ss_data(self):
        self.ss_chain_list.blockSignals(True)
        self.ss_chain_list.value = self.sv.alignment.associations.keys()
        self.ss_chain_list.blockSignals(False)

    def _ss_chain_changed(self):
        self._processing_ss_list = True
        chosen = set(self.ss_chain_list.value)
        for chain in self.ss_chain_list.all_values:
            is_associated = chain in self.sv.alignment.associations
            want_association = chain in chosen
            if is_associated == want_association:
                continue
            from chimerax.core.commands import run
            if want_association:
                run(self.sv.session, "sequence assoc %s%s" % (chain.string(style="command"),
                    self._align_arg()))
            else:
                run(self.sv.session, "sequence disassoc %s%s" % (chain.string(style="command"),
                    self._align_arg()))
        self._processing_ss_list = False
        self._set_ss_data()
