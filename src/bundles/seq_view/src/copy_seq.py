# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

class CopySeqDialog:

    def __init__(self, sv, tool_window):
        self.sv = sv
        self.tool_window = tool_window
        tool_window.help = "help:user/tools/sequenceviewer.html#copy"

        from Qt.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)

        seq_layout = QHBoxLayout()
        seq_layout.setSpacing(3)
        layout.addLayout(seq_layout)

        seq_layout.addWidget(QLabel("Copy"))
        from chimerax.seqalign.widgets import AlignSeqMenuButton
        self.seq_button = AlignSeqMenuButton(sv.alignment, no_value_button_text="(choose sequence)")
        seq_layout.addWidget(self.seq_button)
        seq_layout.addWidget(QLabel("to system paste buffer"))
        from chimerax.ui.options import OptionsPanel, BooleanOption
        options_panel = OptionsPanel(sorting=False, scrolled=False)
        self.remove_gaps_option = BooleanOption("Remove gaps from copy", True, None,
            balloon="Remove gap characters from sequence before copying")
        options_panel.add_option(self.remove_gaps_option)
        self.active_region_option = BooleanOption("Restrict copy to active region", False, None,
            balloon="Restrict the copy to the active region (if any) on the alignment")
        options_panel.add_option(self.active_region_option)
        layout.addWidget(options_panel)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        bbox = qbbox(qbbox.Ok | qbbox.Close | qbbox.Help)
        bbox.accepted.connect(self.copy_seq)
        hide_self = lambda *args, tw=tool_window: setattr(tool_window, 'shown', False)
        bbox.rejected.connect(hide_self)
        from chimerax.core.commands import run
        bbox.helpRequested.connect(lambda *, run=run, ses=self.sv.session, help=tool_window.help:
            run(ses, "help " + help))
        layout.addWidget(bbox)

        tool_window.ui_area.setLayout(layout)

    def copy_seq(self):
        from chimerax.core.errors import UserError
        seq = self.seq_button.value
        if seq is None:
            raise UserError("No sequence chosen for copying")
        if self.active_region_option.value:
            cur_region = self.sv.region_manager.cur_region()
            if cur_region is None:
                raise UserError("No active region")
            get_index = self.sv.alignment.seqs.index
            seq_index = get_index(seq)
            text = ""
            for line1, line2, pos1, pos2 in cur_region.blocks:
                i1 = get_index(line1)
                if seq_index < i1:
                    continue
                i2 = get_index(line2)
                if seq_index > i2:
                    continue
                if self.remove_gaps_option.value:
                    text += "".join([seq[p] for p in range(pos1, pos2+1)
                        if seq.gapped_to_ungapped(p) is not None])
                else:
                    text += seq[pos1:pos2+1]
            if not text:
                raise UserError("Active region does not cover any %scharacters of %s" % (
                    "non-gap " if self.remove_gaps_option.value else "", seq.name))
        elif self.remove_gaps_option.value:
            text = seq.ungapped()
        else:
            text = seq.characters
        self.sv.session.ui.clipboard().setText(text)
        self.tool_window.shown = False
