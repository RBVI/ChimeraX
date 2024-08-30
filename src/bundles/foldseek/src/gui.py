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
class FoldseekPanel(ToolInstance):
    help = 'help:user/tools/foldseek.html'

    def __init__(self, session, tool_name = 'Foldseek'):

        self.results = None
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

        # Results table
        self._results_table = None
        self._results_table_position = layout.count()

        # Options panel
        options = self._create_options_pane(parent)
        layout.addWidget(options)

        # Buttons
        from chimerax.ui.widgets import button_row
        bf = button_row(parent,
                        [('Search', self._search),
                         ('Open', self._open_selected),
                         ('Sequences', self._show_sequences),
                         ('Traces', self._show_backbone_traces),
                         ('Clusters', self._show_cluster_plot),
                         ('Ligands', self._show_ligands),
                         ('Options', self._show_or_hide_options),
                         ('Help', self._show_help)],
                        spacing = 2)
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
    @property
    def hits(self):
        return self.results.hits

    # ---------------------------------------------------------------------------
    #
    def _create_chain_and_database_menus(self, parent):
        from chimerax.atomic.widgets import ChainMenuButton
        from chimerax.atomic import Residue
        cmenu = ChainMenuButton(self.session, parent=parent,
                                filter_func=lambda c: c.polymer_type == Residue.PT_AMINO,
                                no_value_button_text="No chain chosen")

        from chimerax.ui.widgets import EntriesRow
        from .search import foldseek_databases
        fd = EntriesRow(parent, 'Query chain', cmenu, 'from database', tuple(foldseek_databases))
        dbmenu = fd.values[0]
        
        return cmenu, dbmenu, fd.frame

    # ---------------------------------------------------------------------------
    #
    def _create_results_table(self, parent):
        r = self.results
        database = r.database
        if database.startswith('pdb'):
            database_name = 'PDB'
        elif database.startswith('afdb'):
            database_name = 'AFDB'
        else:
            database_name = 'Id'
        fr = FoldseekResultsTable(r.hits, database_name, parent = parent)
        return fr
    
    # ---------------------------------------------------------------------------
    #
    def _create_options_pane(self, parent, trim = True, alignment_cutoff_distance = 2.0):

        from chimerax.ui.widgets import CollapsiblePanel
        self._options_panel = p = CollapsiblePanel(parent, title = None)
        f = p.content_area

        # Whether to delete extra chains and extra residues when loading structures.
        from chimerax.ui.widgets import EntriesRow
        tr = EntriesRow(f, 'Trim', True, 'extra chains,', True, 'sequence ends,', True, 'and far ligands')
        self._trim_extra_chains, self._trim_sequences, self._trim_ligands = tr.values
        for cb in tr.values:
            cb.changed.connect(self._trim_changed)
        self.set_trim_options(trim)

        # For pruning aligned residues when opening hits and aligning them to query chain
        pd = EntriesRow(f, 'Alignment pruning C-alpha atom distance', 2.0)
        self._alignment_cutoff_distance = acd = pd.values[0]
        self.set_alignment_cutoff_option(alignment_cutoff_distance)
        acd.return_pressed.connect(self._alignment_cutoff_distance_changed)
        return p

    # ---------------------------------------------------------------------------
    #
    def _show_or_hide_options(self):
        self._options_panel.toggle_panel_display()

    # ---------------------------------------------------------------------------
    #
    def set_trim_options(self, trim):
        if trim is None:
            return
        if trim in (True, False):
            self._trim_extra_chains.value = self._trim_sequences.value = self._trim_ligands.value = trim
        elif isinstance(trim, (list, tuple)):
            self._trim_extra_chains.value = ('chains' in trim)
            self._trim_sequences.value = ('sequence' in trim)
            self._trim_ligands.value = ('ligands' in trim)
        self._trim_changed()
        
    # ---------------------------------------------------------------------------
    #
    @property
    def trim(self):
        c, s, l = self._trim_extra_chains.value, self._trim_sequences.value, self._trim_ligands.value
        trim = [name for t, name in ((c, 'chains'), (s, 'sequence'), (l, 'ligands')) if t]
        if len(trim) == 3: trim = True
        elif len(trim) == 0: trim = False
        return trim
            
    # ---------------------------------------------------------------------------
    #
    def _trim_changed(self):
        r = self.results
        if r:
            r.trim = self.trim

    # ---------------------------------------------------------------------------
    #
    @property
    def alignment_cutoff_distance(self):
        return self._alignment_cutoff_distance.value
    def set_alignment_cutoff_option(self, alignment_cutoff_distance):
        if alignment_cutoff_distance is not None:
            self._alignment_cutoff_distance.value = alignment_cutoff_distance
            self._alignment_cutoff_distance_changed()
    def _alignment_cutoff_distance_changed(self):
        r = self.results
        if r:
            r.alignment_cutoff_distance = self.alignment_cutoff_distance

    # ---------------------------------------------------------------------------
    #
    def _search(self):
        chain = self._chain_menu.value
        if chain is None:
            self.session.logger.error('Must choose a chain in the Foldseek panel before running search')
            return
        db = self._database_menu.value
        cmd = f'foldseek {chain.string(style="command")}'
        if db != 'pdb100':
            cmd += f' database {db}'
        from chimerax.core.commands import run
        run(self.session, cmd)

    # ---------------------------------------------------------------------------
    #
    def show_results(self, results):
        self.results = results
        results.trim = self.trim
        results.alignment_cutoff_distance = self.alignment_cutoff_distance
        self._chain_menu.value = results.query_chain
        self._database_menu.value = results.database
        rt = self._results_table
        parent = self.tool_window.ui_area
        layout = parent.layout()
        if rt:
            layout.removeWidget(rt)
            rt.deleteLater()
        results.compute_rmsds(self.alignment_cutoff_distance)
        self._results_table = rt = self._create_results_table(parent)
        layout.insertWidget(self._results_table_position, rt)
        self._show_hit_count()
            
    # ---------------------------------------------------------------------------
    #
    def _show_hit_count(self):
        r = self.results
        heading = f'Foldseek search found {r.num_hits} {r.database} hits'
        if r.query_chain:
            q = r.query_chain.string(include_structure = True)
            heading += f' similar to {q}'
        self._heading.setText(heading)
            
    # ---------------------------------------------------------------------------
    #
    def _open_selected(self):
        results_table = self._results_table
        if results_table is None:
            msg = 'You must press the Foldseek Search button before you can open matching structures.'
            self.session.logger.error(msg)
            return
        hit_rows = results_table.selected		# FoldseekRow instances
        for hit_row in hit_rows:
            self.open_hit(hit_row.hit)
        if len(hit_rows) == 0:
            msg = 'Click lines in the Foldseek results table and then press Open.'
            self.session.logger.error(msg)

    # ---------------------------------------------------------------------------
    #
    def open_hit(self, hit):
        from .foldseek import open_hit
        open_hit(self.session, hit, self.results.query_chain, trim = self.trim,
                 alignment_cutoff_distance = self.alignment_cutoff_distance)

    # ---------------------------------------------------------------------------
    #
    def select_table_row(self, hit_or_row_number):
        t = self._results_table
        if isinstance(hit_or_row_number, int):
            row = hit_or_row_number
        else:
            hit = hit_or_row_number
            row = [r for r,h in enumerate(self.results.hits) if h is hit][0]
        item = t.data[row]
        t.selected = [item]
        t.scroll_to(item)

    # ---------------------------------------------------------------------------
    #
    def _show_sequences(self):
        from chimerax.core.commands import run
        run(self.session, 'foldseek sequences')

    # ---------------------------------------------------------------------------
    #
    def _show_backbone_traces(self):
        from chimerax.core.commands import run
        run(self.session, 'foldseek traces')

    # ---------------------------------------------------------------------------
    #
    def _show_cluster_plot(self, *, nres = 5):
        r = self.results
        if r.query_chain is None:
            self.session.logger.error('Cannot compute Foldseek clusters without query structure')
            return
        r.set_conservation_attribute()
        r.set_coverage_attribute()
        cr = [(res.foldseek_conservation * res.foldseek_coverage, res) for res in r.query_residues]
        cr.sort(reverse = True)
        most_conserved_res = [res for c,res in cr[0:nres]]
        rnums = ','.join(str(res.number) for res in most_conserved_res)
        cspec = r.query_chain.string(style = 'command')
        rspec = cspec + f':{rnums}'

        from chimerax.core.commands import run
        run(self.session, f'foldseek cluster {rspec} clusterDistance 1.5')

    # ---------------------------------------------------------------------------
    #
    def _show_ligands(self):
        if self.results.database != 'pdb100':
            self.session.logger.error('Only Foldseek results from the Protein Databank have ligands')
            return

        nhits = self.results.num_hits
        message = f'This will fetch {nhits} PDB structures and align their ligands to the query structure.  It may take several minutes to fetch those structures during which ChimeraX will be frozen.  Do you want to proceed?'
        from chimerax.ui.ask import ask
        answer = ask(self.session, message, title='Fetch Foldseek ligands')
        if answer == 'no':
            return
        
        from chimerax.core.commands import run
        run(self.session, 'foldseek ligands')

    # ---------------------------------------------------------------------------
    #
    def _show_help(self):
        from chimerax.core.commands import run
        run(self.session, 'help %s' % self.help)
    
    # ---------------------------------------------------------------------------
    # Session saving.
    #
    @property
    def SESSION_SAVE(self):
        return len(self.hits) > 0

    def take_snapshot(self, session, flags):
        data = {'results': self.results,
                'trim': self.trim,
                'alignment_cutoff_distance': self.alignment_cutoff_distance,
                'version': '1'}
        return data

    # ---------------------------------------------------------------------------
    # Session restore
    #
    @classmethod
    def restore_snapshot(cls, session, data):
        fp = foldseek_panel(session, create = True)
        fp.show_results(data['hits'], data['query_chain'], data['database'])
        fp.set_trim_options(data['trim'])
        fp.set_alignment_cutoff_option(data['alignment_cutoff_distance'])
        return fp

# -----------------------------------------------------------------------------
#
from chimerax.ui.widgets import ItemTable
class FoldseekResultsTable(ItemTable):
    def __init__(self, hits, database_name, parent = None):
        ItemTable.__init__(self, parent = parent)
        self.add_column(database_name, 'database_full_id')
        col_identity = self.add_column('Identity', 'pident')
        col_evalue = self.add_column('E-value', 'evalue', format = '%.2g')
        if hits and hits[0]:
            if 'close' in hits[0]:
                col_close = self.add_column('% Close', 'close', format = '%.0f')
            if 'coverage' in hits[0]:
                col_coverage = self.add_column('% Cover', 'coverage', format = '%.0f')
        col_species = self.add_column('Species', 'taxname')
        self.add_column('Description', 'description', justification = 'left')
        rows = [FoldseekRow(hit) for hit in hits]
        self.data = rows
        self.launch()
        self.sort_by(col_evalue, self.SORT_ASCENDING)
        col_species_index = self.columns.index(col_species)
        species_column_width = 120
        self.setColumnWidth(col_species_index, species_column_width)
        self.setAutoScroll(False)  # Otherwise click on Description column scrolls horizontally
        from Qt.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)  # Don't resize whole panel width

# -----------------------------------------------------------------------------
#
class FoldseekRow:
    def __init__(self, hit):
        self.hit = hit
    def __getattr__(self, attribute_name):
        return self.hit.get(attribute_name)

# -----------------------------------------------------------------------------
#
def foldseek_scroll_to(session, hit_name):
    '''Show table row for this hit.'''
    from .foldseek import foldseek_hit_by_name
    hit, results = foldseek_hit_by_name(session, hit_name)
    if hit:
        panel = foldseek_panel(session)
        if panel:
            panel.select_table_row(hit)

# -----------------------------------------------------------------------------
#
def show_foldseek_results(session, results):
    msg = f'Foldseek search for similar structures to {results.query_chain} in {results.database} found {len(results.hits)} hits'
    session.logger.info(msg)

    fp = foldseek_panel(session, create = True)
    fp.set_trim_options(results.trim)
    fp.set_alignment_cutoff_option(results.alignment_cutoff_distance)
    fp.show_results(results)
    return fp

# -----------------------------------------------------------------------------
#
def register_foldseek_scrollto_command(logger):
    from chimerax.core.commands import CmdDesc, register, StringArg
    desc = CmdDesc(
        required = [('hit_name', StringArg)],
        synopsis = 'Show Foldseek result table row'
    )
    register('foldseek scrollto', desc, foldseek_scroll_to, logger=logger)

# -----------------------------------------------------------------------------
#
def foldseek_panel(session, create = False):
    return FoldseekPanel.get_singleton(session, create=create)
  
# -----------------------------------------------------------------------------
#
def show_foldseek_panel(session):
    return foldseek_panel(session, create = True)
