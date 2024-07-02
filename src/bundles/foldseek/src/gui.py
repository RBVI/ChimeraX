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
class FoldseekResults(ToolInstance):

    name = 'Foldseek Results'
    help = 'help:user/tools/foldseek.html'

    def __init__(self, session, tool_name = 'Foldseek Results',
                 query_chain = None, hits = [], database = 'pdb100',
                 trim = True, alignment_cutoff_distance = 2.0):
        self._query_chain = query_chain
        self._database = database

        # Whether to delete extra chains and extra residues when loading structures.
        self._trim = trim   # bool or 'chains' or 'sequence'.

        # For pruning aligned residues when opening hits and aligning them to query chain
        self._alignment_cutoff_distance = alignment_cutoff_distance

        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        self.tool_window = tw = MainToolWindow(self)
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(parent, margins = (5,0,5,0))

        from chimerax.ui.widgets import EntriesRow
        q = query_chain.string(include_structure = True)
        heading = EntriesRow(parent, f'Foldseek search found {len(hits)} {database} hits similar to {q}')
        layout.addWidget(heading.frame)

        if database.startswith('pdb'):
            database_name = 'PDB'
        elif database.startswith('afdb'):
            database_name = 'AFDB'
        else:
            database_name = 'Id'
        self._table = FoldseekResultsTable(hits, database_name, parent = parent)
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
        hits = self._table.selected		# FoldseekRow instances
        for hit in hits:
            self._open_hit(hit)

    def _open_hit(self, row):
        from .foldseek import open_hit
        open_hit(self.session, row.hit, self._query_chain, trim = self._trim,
                 alignment_cutoff_distance = self._alignment_cutoff_distance)

    def _show_help(self):
        from chimerax.core.commands import run
        run(self.session, 'help %s' % self.help)

from chimerax.ui.widgets import ItemTable
class FoldseekResultsTable(ItemTable):
    def __init__(self, hits, database_name, parent = None):
        ItemTable.__init__(self, parent = parent)
        self.add_column(database_name, 'database_full_id')
        col_identity = self.add_column('Identity', 'pident')
        self.add_column('Description', 'description', justification = 'left')
        rows = [FoldseekRow(hit) for hit in hits]
        self.data = rows
        self.launch()
        self.sort_by(col_identity, self.SORT_DESCENDING)

class FoldseekRow:
    def __init__(self, hit):
        self.hit = hit
    def __getattr__(self, attribute_name):
        return self.hit.get(attribute_name)
