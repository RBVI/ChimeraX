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

from chimerax.ui import tool_user_error

class MotifTool:

    def __init__(self, grid, tool_window):
        self.grid = grid
        self.tool_window = tool_window
        #tool_window.help = "help:user/tools/profilegrid.html#gridlabel"

        from Qt.QtWidgets import (QVBoxLayout, QLabel, QHBoxLayout, QGridLayout, QRadioButton, QGroupBox,
            QLineEdit, QSpinBox, QDoubleSpinBox)
        from Qt.QtCore import Qt
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.addWidget(QLabel("Choose consecutive cells that meet these criteria"),
            alignment=Qt.AlignCenter)

        group = QGroupBox("Choose")
        layout.addWidget(group)
        motif_layout = QVBoxLayout()
        motif_layout.setSpacing(2)
        motif_layout.setContentsMargins(2,2,2,2)
        group.setLayout(motif_layout)
        seq_layout = QGridLayout()
        seq_layout.setSpacing(2)
        motif_layout.addLayout(seq_layout)
        self.seq_button = QRadioButton("pattern:")
        seq_layout.addWidget(self.seq_button, 0, 0)
        self.seq_entry = QLineEdit(grid.pg.settings.motif_sequence)
        self.seq_entry.editingFinished.connect(lambda seq_but=self.seq_button: seq_but.setChecked(True))
        seq_layout.addWidget(self.seq_entry, 0, 1)
        seq_layout.setColumnStretch(2, 1)
        explanation = QLabel("Multiple residue choices can be listed in brackets, and excluded residues"
            " in curly braces.  A '.' (period) matches any character.")
        explanation.setWordWrap(True)
        from chimerax.ui import shrink_font
        shrink_font(explanation)
        seq_layout.addWidget(explanation, 1, 0, 1, 3)
        cell_layout = QHBoxLayout()
        cell_layout.setSpacing(2)
        motif_layout.addLayout(cell_layout)
        stretch_button = QRadioButton("")
        cell_layout.addWidget(stretch_button)
        self.stretch_len = QSpinBox()
        self.stretch_len.setRange(1, 99)
        self.stretch_len.setValue(grid.pg.settings.motif_length)
        cell_layout.addWidget(self.stretch_len)
        cell_layout.addWidget(QLabel("consecutive non-gap cells"))
        cell_layout.addStretch(1)
        if grid.pg.settings.motif_type == "stretch":
            checked_button = stretch_button
        else:
            checked_button = self.seq_button
        checked_button.setChecked(True)

        percent_layout = QHBoxLayout()
        percent_layout.setSpacing(2)
        layout.addLayout(percent_layout)
        percent_layout.addStretch(1)
        percent_layout.addWidget(QLabel("where each cell has"))
        self.percent_box = QDoubleSpinBox()
        self.percent_box.setRange(0.0, 100.0)
        self.percent_box.setSuffix('%')
        self.percent_box.setDecimals(1)
        self.percent_box.setValue(grid.pg.settings.motif_percentage)
        percent_layout.addWidget(self.percent_box)
        percent_layout.addWidget(QLabel("occupancy or higher"))
        percent_layout.addStretch(1)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        self.bbox = bbox = qbbox(qbbox.Ok | qbbox.Apply | qbbox.Close | qbbox.Help)
        bbox.accepted.connect(self.find_motif)
        bbox.rejected.connect(lambda tw=self.tool_window: setattr(tw, 'shown', False))
        # Since ApplyRole is not AcceptRole, simply connecting to the Apply button won't dismiss the dialog
        bbox.button(qbbox.Apply).clicked.connect(lambda *args, fm=self.find_motif: fm(apply=True))
        if getattr(tool_window, 'help', None) is None:
            bbox.button(qbbox.Help).setEnabled(False)
        else:
            from chimerax.core.commands import run
            bbox.helpRequested.connect(lambda *, run=run, ses=grid.pg.session, tw=tool_window:
                run(ses, "help " + tw.help))
        layout.addWidget(bbox)

        tool_window.ui_area.setLayout(layout)

    def find_motif(self, *, apply=False):
        from chimerax.ui import tool_user_error
        if not apply:
            self.tool_window.shown = False
            self.grid.pg.session.ui.processEvents()

        motif = []
        any_char = set([c for c in self.grid.existing_row_labels if len(c) == 1])
        do_seq = self.seq_button.isChecked()
        if do_seq:
            seq = self.seq_entry.text().strip().upper()
            if not seq:
                self.tool_window.shown = True
                return tool_user_error("No sequence specified")
            motif_state = "open"
            seq_index = 0
            while seq_index < len(seq):
                char = seq[seq_index]
                if motif_state == "open":
                    if char.isalpha():
                        motif.append(set(char))
                    elif char == '.':
                        motif.append(any_char)
                    elif char == '{':
                        chars = set()
                        motif_state = '}'
                    elif char == '[':
                        chars = set()
                        motif_state = ']'
                    else:
                        self.tool_window.shown = True
                        return tool_user_error(f"Unexpected character in sequence: '{char}'")
                elif motif_state == ']':
                    if char.isalpha():
                        chars.add(char)
                    elif char in '.{}[':
                        self.tool_window.shown = True
                        return tool_user_error(f"'{char}' not allowed inside brackets")
                    elif char == ']':
                        if not chars:
                            self.tool_window.shown = True
                            return tool_user_error(f"Empty brackets not allowed")
                        motif.append(chars)
                        motif_state = "open"
                    else:
                        self.tool_window.shown = True
                        return tool_user_error(f"Unexpected character in sequence: '{char}'")
                elif motif_state == '}':
                    if char.isalpha():
                        chars.add(char)
                    elif char in '.{][':
                        self.tool_window.shown = True
                        return tool_user_error(f"'{char}' not allowed inside curly braces")
                    elif char == '}':
                        if not chars:
                            self.tool_window.shown = True
                            return tool_user_error(f"Empty curly braces not allowed")
                        motif.append(any_char - chars)
                        motif_state = "open"
                    else:
                        self.tool_window.shown = True
                        return tool_user_error(f"Unexpected character in sequence: '{char}'")
                seq_index += 1
            if motif_state != "open":
                self.tool_window.shown = True
                return tool_user_error(f"Sequence has no closing '{motif_state}' character")
            self.grid.pg.settings.motif_sequence = seq
        else:
            # stretch
            stretch_len = self.stretch_len.value()
            for i in range(stretch_len):
                motif.append(any_char)
            self.grid.pg.settings.motif_length = stretch_len

        percent = self.percent_box.value()
        target_fraction = percent / 100.0
        divisor = sum(self.grid.weights)

        matches = []
        for aln_index in range(len(self.grid.alignment.seqs[0]) - len(motif) + 1):
            match = []
            for motif_index, motif_chars in enumerate(motif):
                col_index = aln_index + motif_index
                col_matched = False
                for row, row_label in enumerate(self.grid.row_labels):
                    if row_label not in motif_chars:
                        continue
                    if self.grid.grid_data[row,col_index] / divisor < target_fraction:
                        continue
                    match.append((row_label, col_index))
                    col_matched = True
                if not col_matched:
                    match = None
                    break
            if match is not None:
                matches.append(match)

        from chimerax.core.commands import plural_form
        # since choosing the cells will report number of sequences matches in primary status, use secondary
        self.grid.pg.status("Found %d %s to the motif" % (len(matches), plural_form(matches, "match")),
            secondary=True)

        chosen_cells = []
        label_to_row = { l:r for r, l in enumerate(self.grid.existing_row_labels) }
        for match in matches:
            chosen_cells.extend([(label_to_row[label], col) for label, col in match])
        self.grid._choose_cells(chosen_cells)

        self.grid.pg.settings.motif_type = "sequence" if do_seq else "stretch"
        self.grid.pg.settings.motif_percentage = percent
