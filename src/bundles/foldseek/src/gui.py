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

# -----------------------------------------------------------------------------
# Panel for choosing structure and database for Foldseek search.
#
from chimerax.core.tools import ToolInstance
class Foldseek(ToolInstance):
    help = 'help:user/tools/foldseek.html'

    def __init__(self, session, tool_name = 'Foldseek',
                 query_chain = None, database = None,
                 hits = [], trim = True, alignment_cutoff_distance = 2.0):
        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        self.tool_window = tw
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(parent, margins = (5,0,0,0))

        from chimerax.ui.widgets import EntriesRow
        heading = 'Search for structures with the same fold using Foldseek'
        hl = EntriesRow(parent, heading)
        self._heading = hl.labels[0]
        layout.addWidget(hl.frame)
        
        # Make menus to choose chain and databases
        self._chain_menu, self._database_menu, mf = self._create_chain_and_database_menus(parent)
        layout.addWidget(mf)
        if query_chain is not None:
            self._chain_menu.value = query_chain
        if database is not None:
            self._database_menu.value = database

        # Results table
        self._results_table = None
        self._results_table_position = layout.count()
        self._results_query_chain = None
        if hits:
            self._results_query_chain = query_chain
            self._results_table = rt = self._create_results_table(parent, hits, database)
            layout.addWidget(rt)
            self._show_hit_count(len(hits), query_chain, database)

        # Options panel
        options = self._create_options_pane(parent, trim, alignment_cutoff_distance)
        layout.addWidget(options)

        # Buttons
        from chimerax.ui.widgets import button_row
        bf = button_row(parent,
                        [('Search', self._search),
                         ('Open', self._open_selected),
                         ('Options', self._show_or_hide_options),
                         ('Help', self._show_help)],
                        spacing = 10)
        bf.setContentsMargins(0,5,0,0)
        layout.addWidget(bf)

        layout.addStretch(1)    # Extra space at end

        tw.manage(placement="side")

    # ---------------------------------------------------------------------------
    #
    @classmethod
    def get_singleton(cls, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, cls, 'Foldseek', create=create)
    
    # ---------------------------------------------------------------------------
    #
    def _create_chain_and_database_menus(self, parent):
        from chimerax.atomic.widgets import ChainMenuButton
        from chimerax.atomic import Residue
        cmenu = ChainMenuButton(self.session, parent=parent,
                                filter_func=lambda c: c.polymer_type == Residue.PT_AMINO,
                                no_value_button_text="No chain chosen")

        from chimerax.ui.widgets import EntriesRow
        from .foldseek import foldseek_databases
        fd = EntriesRow(parent, 'Query chain', cmenu, 'from database', tuple(foldseek_databases))
        dbmenu = fd.values[0]
        
        return cmenu, dbmenu, fd.frame

    # ---------------------------------------------------------------------------
    #
    def _create_results_table(self, parent, hits, database):
        if database.startswith('pdb'):
            database_name = 'PDB'
        elif database.startswith('afdb'):
            database_name = 'AFDB'
        else:
            database_name = 'Id'
        fr = FoldseekResultsTable(hits, database_name, parent = parent)
        return fr
    
    # ---------------------------------------------------------------------------
    #
    def _create_options_pane(self, parent, trim, alignment_cutoff_distance):

        from chimerax.ui.widgets import CollapsiblePanel
        self._options_panel = p = CollapsiblePanel(parent, title = None)
        f = p.content_area

        # Whether to delete extra chains and extra residues when loading structures.
        from chimerax.ui.widgets import EntriesRow
        tr = EntriesRow(f, 'Trim', True, 'extra chains,', True, 'sequence ends,', True, 'and far ligands')
        self._trim_extra_chains, self._trim_sequences, self._trim_ligands = tr.values
        self._set_trim_options(trim)

        # For pruning aligned residues when opening hits and aligning them to query chain
        pd = EntriesRow(f, 'Alignment pruning C-alpha atom distance', 2.0)
        self._alignment_cutoff_distance = pd.values[0]
        if alignment_cutoff_distance is not None:
            self._alignment_cutoff_distance.value = alignment_cutoff_distance

        return p

    # ---------------------------------------------------------------------------
    #
    def _show_or_hide_options(self):
        self._options_panel.toggle_panel_display()

    # ---------------------------------------------------------------------------
    #
    def _set_trim_options(self, trim):
        if trim is None:
            return
        if trim in (True, False):
            self._trim_extra_chains.value = self._trim_sequences.value = self._trim_ligands.value = trim
        elif isinstance(trim, (list, tuple)):
            self._trim_extra_chains.value = ('chains' in trim)
            self._trim_sequences.value = ('sequence' in trim)
            self._trim_ligands.value = ('ligands' in trim)
        
    # ---------------------------------------------------------------------------
    #
    @property
    def _trim(self):
        c, s, l = self._trim_extra_chains.value, self._trim_sequences.value, self._trim_ligands.value
        trim = [name for t, name in ((c, 'chains'), (s, 'sequence'), (l, 'ligands')) if t]
        if len(trim) == 3: trim = True
        elif len(trim) == 0: trim = False
        return trim

    # ---------------------------------------------------------------------------
    #
    def _search(self):
        chain = self._chain_menu.value
        if chain is None:
            self.session.logger.warning('Must choose a chain in the Foldseek panel before running search')
        db = self._database_menu.value
        cmd = f'foldseek {chain.string(style="command")}'
        if db != 'pdb100':
            cmd += f' database {db}'
        from chimerax.core.commands import run
        run(self.session, cmd)

    # ---------------------------------------------------------------------------
    #
    def show_results(self, hits, query_chain, database, trim = None, alignment_cutoff_distance = None):
        self._chain_menu.value = query_chain
        self._results_query_chain = query_chain
        self._database_menu.value = database
        self._set_trim_options(trim)
        if alignment_cutoff_distance is not None:
            self._alignment_cutoff_distance.value = alignment_cutoff_distance
        rt = self._results_table
        parent = self.tool_window.ui_area
        layout = parent.layout()
        if rt:
            layout.removeWidget(rt)
            rt.destroy()
        self._results_table = rt = self._create_results_table(parent, hits, database)
        layout.insertWidget(self._results_table_position, rt)
        self._show_hit_count(len(hits), query_chain, database)
            
    # ---------------------------------------------------------------------------
    #
    def _show_hit_count(self, nhits, query_chain, database):
        q = query_chain.string(include_structure = True)
        heading = f'Foldseek search found {nhits} {database} hits similar to {q}'
        self._heading.setText(heading)
            
    # ---------------------------------------------------------------------------
    #
    def _open_selected(self):
        hits = self._results_table.selected		# FoldseekRow instances
        for hit in hits:
            self._open_hit(hit)

    # ---------------------------------------------------------------------------
    #
    def _open_hit(self, row):
        from .foldseek import open_hit
        open_hit(self.session, row.hit, self._results_query_chain, trim = self._trim,
                 alignment_cutoff_distance = self._alignment_cutoff_distance.value)

    # ---------------------------------------------------------------------------
    #
    def _show_help(self):
        from chimerax.core.commands import run
        run(self.session, 'help %s' % self.help)

# -----------------------------------------------------------------------------
#
from chimerax.ui.widgets import ItemTable
class FoldseekResultsTable(ItemTable):
    def __init__(self, hits, database_name, parent = None):
        ItemTable.__init__(self, parent = parent)
        self.add_column(database_name, 'database_full_id')
        col_identity = self.add_column('Identity', 'pident')
        col_species = self.add_column('Species', 'taxname')
        self.add_column('Description', 'description', justification = 'left')
        rows = [FoldseekRow(hit) for hit in hits]
        self.data = rows
        self.launch()
        self.sort_by(col_identity, self.SORT_DESCENDING)
        col_species_index = self.columns.index(col_species)
        species_column_width = 120
        self.setColumnWidth(col_species_index, species_column_width)
        self.setAutoScroll(False)  # Otherwise click on Description column scrolls horizontally

# -----------------------------------------------------------------------------
#
class FoldseekRow:
    def __init__(self, hit):
        self.hit = hit
    def __getattr__(self, attribute_name):
        return self.hit.get(attribute_name)

# -----------------------------------------------------------------------------
#
def foldseek_panel(session, create = False):
    return Foldseek.get_singleton(session, create=create)
  
# -----------------------------------------------------------------------------
#
def show_foldseek_panel(session):
    return foldseek_panel(session, create = True)
