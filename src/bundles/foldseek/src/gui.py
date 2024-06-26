# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
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

from chimerax.core.tools import ToolInstance
class FoldseekPDBResults(ToolInstance):

    name = 'Foldseek Results'
    help = 'help:user/tools/foldseek.html'

    def __init__(self, session, tool_name = 'Foldseek Results',
                 query_structure = None, pdb_hits = [], trim = True):
        self._query_structure = query_structure
        self._trim = trim   # bool or 'chains' or 'sequence'.  Whether to delete extra chains and extra residues when loading structures.

        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(parent, margins = (5,0,5,0))

        from chimerax.ui.widgets import EntriesRow
        heading = EntriesRow(parent, f'Foldseek search found {len(pdb_hits)} PDB hits')
        layout.addWidget(heading.frame)

        self._table = FoldseekPDBResultsTable(pdb_hits, parent = parent)
        layout.addWidget(self._table)

        from chimerax.ui.widgets import button_row
        bf = button_row(parent,
                        [('Open', self._open_selected),
                         ('Help', self._show_help)],
                        spacing = 10)
        bf.setContentsMargins(0,5,0,0)
        layout.addWidget(bf)

        layout.addStretch(1)    # Extra space at end

        tw.manage(placement=None)	# Start floating

    def _open_selected(self):
        pdb_hits = self._table.selected		# FoldseekPDBRow instances
        for hit in pdb_hits:
            self._open_pdb_hit(hit)

    def _open_pdb_hit(self, pdb_row):
        from .foldseek import open_pdb_hit
        open_pdb_hit(self.session, pdb_row.pdb_hit, self._query_structure, trim = self._trim)

    def _show_help(self):
        from chimerax.core.commands import run
        run(self.session, 'help %s' % self.help)

from chimerax.ui.widgets import ItemTable
class FoldseekPDBResultsTable(ItemTable):
    def __init__(self, pdb_hits, parent = None):
        ItemTable.__init__(self, parent = parent)
        self.add_column('PDB', 'pdb_id_and_chain_id')
        self.add_column('Description', 'pdb_description', justification = 'left')
        rows = [FoldseekPDBRow(hit) for hit in pdb_hits]
        self.data = rows
        self.launch()

class FoldseekPDBRow:
    def __init__(self, pdb_hit):
        self.pdb_hit = pdb_hit
    def __getattr__(self, attribute_name):
        return self.pdb_hit.get(attribute_name)
    @property
    def pdb_id_and_chain_id(self):
        return f'{self.pdb_id}_{self.pdb_chain_id}'
