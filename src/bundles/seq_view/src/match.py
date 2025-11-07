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

class MatchDialog:
    def __init__(self, sv, tool_window):
        self.sv = sv
        self.tool_window = tool_window
        tool_window.help = "help:user/tools/sequenceviewer.html#match"

        from Qt.QtWidgets import QHBoxLayout, QVBoxLayout, QLabel, QGridLayout, QLineEdit, QCheckBox
        from Qt.QtGui import QDoubleValidator
        from Qt.QtCore import Qt
        layout = QVBoxLayout()
        #layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)

        from chimerax.atomic.widgets import ChainMenuButton, ChainListWidget
        chains_layout = QGridLayout()
        chains_layout.setSpacing(3)
        chains_layout.addWidget(QLabel("Reference chain"), 0, 0, alignment=Qt.AlignCenter)
        chains_layout.addWidget(QLabel("Match chain(s)"), 0, 1, alignment=Qt.AlignCenter)
        self.ref_chain_menu = ChainMenuButton(sv.session, list_func=sv.alignment.associations.keys)
        chains_layout.addWidget(self.ref_chain_menu, 1, 0, alignment=Qt.AlignHCenter|Qt.AlignTop)
        self.match_chain_list = ChainListWidget(sv.session, autoselect=ChainListWidget.AUTOSELECT_FIRST,
            list_func=sv.alignment.associations.keys, filter_func=lambda c, menu=self.ref_chain_menu:
            c.structure is not getattr(menu.value, 'structure', None))
        chains_layout.addWidget(self.match_chain_list, 1, 1)
        self.ref_chain_menu.value_changed.connect(self.match_chain_list.refresh)
        chains_layout.setRowStretch(1, 1)
        layout.addLayout(chains_layout)

        cv_layout = QHBoxLayout()
        self.conserved_check_box = QCheckBox("Only match residues in columns with at least ")
        cv_layout.addWidget(self.conserved_check_box)
        self.conservation_value = cv = QLineEdit("80")
        cv.setAlignment(Qt.AlignCenter)
        cv.setMaximumWidth(4 * cv.fontMetrics().averageCharWidth())
        cv.setValidator(QDoubleValidator(0.0, 100.0, -1))
        cv_layout.addWidget(cv)
        cv_layout.addWidget(QLabel("% identity"), alignment=Qt.AlignLeft, stretch=1)
        layout.addLayout(cv_layout)

        it_layout = QHBoxLayout()
        self.iterate_check_box = QCheckBox("Iterate by pruning long atom pairs until no pair exceeds ")
        self.iterate_check_box.setChecked(True)
        it_layout.addWidget(self.iterate_check_box)
        self.iteration_value = iv = QLineEdit("2.0")
        iv.setAlignment(Qt.AlignCenter)
        iv.setMaximumWidth(4 * iv.fontMetrics().averageCharWidth())
        dv = QDoubleValidator()
        dv.setBottom(0.0)
        iv.setValidator(dv)
        it_layout.addWidget(iv)
        it_layout.addWidget(QLabel(" angstroms"), alignment=Qt.AlignLeft, stretch=1)
        layout.addLayout(it_layout)

        ur_layout = QHBoxLayout()
        self.use_region_check_box = QCheckBox("Match active region only")
        ur_layout.addWidget(self.use_region_check_box, alignment=Qt.AlignLeft, stretch=1)
        layout.addLayout(ur_layout)

        cr_layout = QHBoxLayout()
        self.create_region_check_box = QCheckBox("Create region showing matched residues")
        cr_layout.addWidget(self.create_region_check_box, alignment=Qt.AlignLeft, stretch=1)
        layout.addLayout(cr_layout)

        from Qt.QtWidgets import QDialogButtonBox as qbbox
        bbox = qbbox(qbbox.Ok | qbbox.Apply | qbbox.Close | qbbox.Help)
        bbox.accepted.connect(self.match)
        bbox.button(qbbox.Apply).clicked.connect(lambda *args, f=self.match: f(apply=True))
        hide_self = lambda *args, tw=tool_window: setattr(tool_window, 'shown', False)
        bbox.rejected.connect(hide_self)
        from chimerax.core.commands import run
        bbox.helpRequested.connect(lambda *, run=run, ses=self.sv.session, help=tool_window.help:
            run(ses, "help " + help))
        layout.addWidget(bbox)

        tool_window.ui_area.setLayout(layout)

    def match(self, *, apply=False):
        from chimerax.core.errors import UserError
        ref = self.ref_chain_menu.value
        if ref is None:
            raise UserError("No reference chain chosen")
        used_chains = { ref.structure: ref }
        matches = self.match_chain_list.value
        if not matches:
            raise UserError("No match chains chosen")
        for match in matches:
            if match.structure in used_chains:
                self.tool_window.shown = True
                raise UserError("Cannot match multiple chains from the same structure (%s and %s)"
                    % (used_chains[match.structure], match))
            used_chains[match.structure] = match

        args = ""
        if self.conserved_check_box.isChecked():
            if not self.conservation_value.hasAcceptableInput():
                raise UserError("Percent identity value must be between 0 and 100")
            args += " conservation " + self.conservation_value.text()
        if self.iterate_check_box.isChecked():
            if not self.iteration_value.hasAcceptableInput():
                raise UserError("Distance cutoff value must be 0 or more")
            args += " cutoffDistance " + self.iteration_value.text()
        else:
            args += " cutoffDistance none"
        if self.use_region_check_box.isChecked():
            region = self.sv.active_region
            if region is None:
                raise UserError("No active region")
            cols = []
            for block in region.blocks:
                line1, line2, pos1, pos2 = block
                cols.extend(list(range(pos1+1,pos2+2)))
            if not cols:
                raise UserError("Active region is empty")
            args += " columns %s" % ','.join(["%d" % col for col in cols])

        if not apply:
            self.tool_window.shown = False
        from chimerax.core.commands import run, StringArg
        results = run(self.sv.session, "seq match %s %s to %s%s" %
            (StringArg.unparse(self.sv.alignment.ident),
            ''.join([c.atomspec for c in matches]), ref.atomspec, args))

        if self.create_region_check_box.isChecked():
            def get_info(a, aln=self.sv.alignment):
                r = a.residue
                c = r.chain
                aseq = aln.associations[c]
                mmap = aln.match_maps[aseq][c]
                return aseq, aseq.ungapped_to_gapped(mmap[r])
            region_indices = { seq: set() for seq in self.sv.alignment.seqs }
            for match_atoms, ref_atoms, rmsd, full_rmsd, xform in results:
                if match_atoms is None:
                    continue
                for atoms in (match_atoms, ref_atoms):
                    for a in atoms:
                        aseq, index = get_info(a)
                        region_indices[aseq].add(index)
            region_blocks = []
            seqs = self.sv.alignment.seqs
            while seqs:
                aseq = seqs.pop(0)
                aseq_indices = region_indices.pop(aseq)
                if not aseq_indices:
                    continue
                collate(aseq, aseq_indices, region_indices, seqs, region_blocks)
            if not region_blocks:
                self.sv.sesson.logger.warn("No matching occurred; not creating 'match' region")
            reg_name, fill, outline = self.sv.MATCHED_REGION_INFO
            reg = self.sv.region_manager.get_region(reg_name, fill=fill, outline=outline, create=True,
                cover_gaps=True)
            reg.clear()
            reg.add_blocks(region_blocks)

def collate(aseq, aseq_indices, region_indices, seqs, region_blocks):
    while aseq_indices:
        first_i = last_i = min(aseq_indices)
        while last_i+1 in aseq_indices:
            last_i += 1
        first_seq = last_seq = aseq
        for seq in seqs:
            seq_indices = region_indices[seq]
            for i in range(first_i, last_i+1):
                if i not in seq_indices:
                    break
            else:
                last_seq = seq
                continue
            break
        region_blocks.append((first_seq, last_seq, first_i, last_i))
        for i in range(first_i, last_i+1):
            aseq_indices.remove(i)
            if last_seq != aseq:
                for seq in seqs:
                    region_indices[seq].remove(i)
                    if seq == last_seq:
                        break
