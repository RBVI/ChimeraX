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
    not_associated_text = "Not associated"

    def __init__(self, grid, tool_window):
        self.grid = grid
        self.tool_window = tool_window
        tool_window.help = "help:user/tools/sequenceviewer.html#association"

        from Qt.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QWidget, QPushButton, QMenu
        from Qt.QtCore import Qt
        layout = QHBoxLayout()
        layout.setSpacing(2)

        from chimerax.atomic.widgets import ChainListWidget
        self.chain_list = ChainListWidget(grid.pg.session, autoselect=ChainListWidget.AUTOSELECT_SINGLE)
        self.chain_list.value_changed.connect(self._chain_changed)
        layout.addWidget(self.chain_list)

        menu_layout = QVBoxLayout()
        layout.addLayout(menu_layout)

        menu_layout.addStretch(1)

        self.pick_a_chain = QLabel("Choose one or more\nchains from the left")
        menu_layout.addWidget(self.pick_a_chain)

        self.assoc_button_header = QLabel("Associated sequence")
        menu_layout.addWidget(self.assoc_button_header, alignment=Qt.AlignBottom|Qt.AlignHCenter)
        from chimerax.seqalign.widgets import AlignSeqMenuButton

        self.best_assoc_label = "Best-matching sequence"
        self.assoc_button = QPushButton()
        menu = QMenu(self.assoc_button)
        menu.aboutToShow.connect(lambda *, pg=self.grid.pg, m=menu, init=self._init_menu:
            pg._menu_of_seqs(m, "", pg.alignment.seqs, self._seq_changed, initialize_menu_func=init))
        self.assoc_button.setMenu(menu)

        #self.assoc_button.value_changed.connect(self._seq_changed)
        menu_layout.addWidget(self.assoc_button, alignment=Qt.AlignTop|Qt.AlignHCenter)

        menu_layout.addStretch(3)

        tool_window.ui_area.setLayout(layout)

        # get initial assoc info correct
        self._chain_changed()

    def _align_arg(self):
        if len(self.grid.pg.session.alignments) > 1:
            return ' ' + self.grid.alignment.ident
        return ''

    def _assoc_mod(self, note_data):
        # called from sequence viewer if associations modified
        self._chain_changed()

    def _chain_changed(self):
        if self.chain_list.count() == 0:
            self.tool_window.shown = False
        chains = self.chain_list.value
        if len(chains) == 0:
            self.assoc_button.setText(self.not_associated_text)
            show_button = False
        elif len(chains) == 1:
            try:
                assoc_text = self.grid.alignment.associations[chains[0]].name
            except KeyError:
                assoc_text = self.not_associated_text
            self.assoc_button.setText(assoc_text)
            show_button = True
        else:
            values = set([self.grid.alignment.associations.get(chain, None) for chain in chains])
            if len(values) == 1:
                val = values.pop()
                if val is None:
                    assoc_text = self.not_associated_text
                else:
                    assoc_text = val.name
                self.assoc_button.setText(assoc_text)
            else:
                self.assoc_button.setText(self.multiseq_text)
            show_button = True
        self.assoc_button_header.setHidden(not show_button)
        self.assoc_button.setHidden(not show_button)
        self.pick_a_chain.setHidden(show_button)

    def _init_menu(self, menu):
        for label, value in [("(none)", None), ("Best-matching sequence", True)]:
            action = menu.addAction(label)
            action.triggered.connect(lambda *args, ab=self.assoc_button, f=self._seq_changed,
                nat=self.not_associated_text, val=value: (ab.setText(nat), f(val)))
        menu.addSeparator()
        return False  # don't recursively initialize submenus

    def _seq_changed(self, req_assoc):
        for chain in self.chain_list.value:
            cur_assoc = self.grid.alignment.associations.get(chain, None)
            if cur_assoc == req_assoc:
                continue
            from chimerax.core.commands import run
            if not req_assoc:
                run(self.grid.pg.session, "sequence disassoc %s%s" % (chain.string(style="command"),
                    self._align_arg()))
            elif req_assoc is True:
                run(self.grid.pg.session, "sequence assoc %s%s" % (chain.string(style="command"),
                    self._align_arg()))
            else:
                run(self.grid.pg.session, "sequence assoc %s %s:%d" % (chain.string(style="command"),
                    self.grid.alignment.ident, self.grid.alignment.seqs.index(req_assoc)+1))
