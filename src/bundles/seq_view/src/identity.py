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

from chimerax.atomic.seq_support import IdentityDenominator

class PercentIdentityDialog:
    all_seqs = "all sequences"

    def __init__(self, sv, tool_window):
        self.sv = sv
        self.tool_window = tool_window
        tool_window.help = "help:user/tools/sequenceviewer.html#context"

        from Qt.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)

        seq_layout = QHBoxLayout()
        seq_layout.setSpacing(3)
        layout.addLayout(seq_layout)

        from chimerax.ui.options import OptionsPanel, SymbolicEnumOption
        from chimerax.seqalign.widgets import AlignSeqMenuOption
        options_panel = OptionsPanel(sorting=False, scrolled=False)
        self.seq1_option = AlignSeqMenuOption(sv.alignment, "Compare:", self.all_seqs, None,
            special_items=[self.all_seqs])
        options_panel.add_option(self.seq1_option)
        self.seq2_option = AlignSeqMenuOption(sv.alignment, "with:", self.all_seqs, None,
            special_items=[self.all_seqs])
        options_panel.add_option(self.seq2_option)
        class Denominator(SymbolicEnumOption):
            values = [x for x in IdentityDenominator]
            labels = [x.description for x in IdentityDenominator]
        self.denominator_option = Denominator("divide by:", IdentityDenominator.default, None)
        options_panel.add_option(self.denominator_option)
        layout.addWidget(options_panel)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        bbox = qbbox(qbbox.Ok | qbbox.Close | qbbox.Help)
        bbox.accepted.connect(self.report_identity)
        hide_self = lambda *args, tw=tool_window: setattr(tool_window, 'shown', False)
        bbox.rejected.connect(hide_self)
        from chimerax.core.commands import run
        bbox.helpRequested.connect(lambda *, run=run, ses=self.sv.session, help=tool_window.help:
            run(ses, "help " + help))
        layout.addWidget(bbox)

        tool_window.ui_area.setLayout(layout)

    def report_identity(self):
        from chimerax.core.commands import run, StringArg
        denom = self.denominator_option.value
        if denom == IdentityDenominator.default:
            denom_string = ""
        else:
            denom_string = " denom %s" % denom.value
        src1 = self.seq1_option.value
        src2 = self.seq2_option.value
        if src1 == self.all_seqs and src2 == self.all_seqs:
            run(self.sv.session, "seq id %s%s"
                % (StringArg.unparse(self.sv.alignment.ident), denom_string))
        else:
            if len(self.sv.session.alignments.alignments) == 1:
                align_arg = ""
            else:
                align_arg = self.sv.alignment.ident
            if src1 == self.all_seqs:
                arg1 = self.sv.alignment.ident
                arg2 = align_arg + ':' + str(self.sv.alignment.seqs.index(src2)+1)
            elif src2 == self.all_seqs:
                arg1 = self.sv.alignment.ident
                arg2 = align_arg + ':' + str(self.sv.alignment.seqs.index(src1)+1)
            else:
                arg1 = align_arg + ':' + str(self.sv.alignment.seqs.index(src1)+1)
                arg2 = align_arg + ':' + str(self.sv.alignment.seqs.index(src2)+1)
            if src2 == self.all_seqs:
                seqs2 = range(len(self.sv.alignment.seqs))
            else:
                seqs2 = [self.sv.alignment.seqs.index(src2)]
            run(self.sv.session,
                "seq id %s %s%s" % (StringArg.unparse(arg1), StringArg.unparse(arg2), denom_string))
        self.tool_window.shown = False
