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

class RenameSeqDialog:

    def __init__(self, sv, tool_window):
        self.sv = sv
        self.tool_window = tool_window
        tool_window.help = "help:user/tools/sequenceviewer.html#rename"

        from Qt.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QLineEdit
        from Qt.QtCore import Qt
        layout = QVBoxLayout()
        layout.setSpacing(0)

        seq_layout = QHBoxLayout()
        seq_layout.setSpacing(3)
        layout.addLayout(seq_layout)

        seq_layout.addWidget(QLabel("Rename"), alignment=Qt.AlignRight)
        from chimerax.seqalign.widgets import AlignSeqMenuButton
        self.seq_button = AlignSeqMenuButton(sv.alignment, no_value_button_text="(choose sequence)")
        seq_layout.addWidget(self.seq_button, alignment=Qt.AlignLeft)

        name_layout = QHBoxLayout()
        name_layout.setSpacing(3)
        layout.addLayout(name_layout)

        name_layout.addWidget(QLabel("as:"))
        self.name_entry = QLineEdit()
        self.name_entry.returnPressed.connect(self.rename_seq)
        name_layout.addWidget(self.name_entry, stretch=1)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        bbox = qbbox(qbbox.Ok | qbbox.Close | qbbox.Help)
        bbox.accepted.connect(self.rename_seq)
        hide_self = lambda *args, tw=tool_window: setattr(tool_window, 'shown', False)
        bbox.rejected.connect(hide_self)
        from chimerax.core.commands import run
        bbox.helpRequested.connect(lambda *, run=run, ses=self.sv.session, help=tool_window.help:
            run(ses, "help " + help))
        layout.addWidget(bbox)

        tool_window.ui_area.setLayout(layout)

    def rename_seq(self):
        seq = self.seq_button.value
        if seq is None:
            return self._error("No sequence chosen for renaming")
        new_name = self.name_entry.text().strip()
        if not new_name:
            return self._error("No new sequence name provided")
        if ':' in new_name:
            return self._error("New sequence name cannot contain ':' character")
        self.tool_window.shown = False
        from chimerax.core.commands import run, StringArg
        run(self.sv.session, "seq rename %s %s" %
            (StringArg.unparse(self.sv.alignment.ident + ':' + seq.name), StringArg.unparse(new_name)))

    def _error(self, msg):
        from chimerax.ui import tool_user_error
        tool_user_error(msg, self.tool_window.ui_area, title="Rename Dialog Error")
