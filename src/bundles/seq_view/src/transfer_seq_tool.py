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

class TransferSeqTool:

    def __init__(self, sv, tool_window):
        self.sv = sv
        self.tool_window = tool_window
        tool_window.help = "help:user/tools/sequenceviewer.html#association"

        from Qt.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QPushButton
        from Qt.QtCore import Qt
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Replace structure sequence with alignment sequence for chosen chains"))

        from chimerax.atomic.widgets import ChainListWidget
        self.chain_list = ChainListWidget(sv.session, autoselect=ChainListWidget.AUTOSELECT_SINGLE,
            filter_func=lambda chain, sv=sv: chain in sv.alignment.associations, trigger_info=None)
        layout.addWidget(self.chain_list)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        bbox = qbbox(qbbox.Ok | qbbox.Close | qbbox.Help)
        bbox.accepted.connect(self.transfer_seq)
        hide_self = lambda *args, tw=tool_window: setattr(tool_window, 'shown', False)
        bbox.rejected.connect(hide_self)
        from chimerax.core.commands import run
        bbox.helpRequested.connect(lambda *, run=run, ses=self.sv.session, help=tool_window.help:
            run(ses, "help " + help))
        layout.addWidget(bbox)

        tool_window.ui_area.setLayout(layout)

    def transfer_seq(self):
        chains = self.chain_list.value
        if not chains:
            from chimerax.core.errors import UserError
            raise UserError("No chains chosen from list")
        from chimerax.core.commands import run
        align_arg = "" if len(self.sv.session.alignments) == 1 else " alignment " + self.sv.alignment.ident
        run(self.sv.session, "seq update " + ''.join([chain.atomspec for chain in chains]) + align_arg)
        self.tool_window.shown = False

    def _align_arg(self):
        if len(self.sv.session.alignments) > 1:
            return ' ' + self.sv.alignment.ident
        return ''

    def _assoc_mod(self, note_data):
        # called from sequence viewer if associations modified
        self.chain_list.refresh()
