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

from chimerax.core.tools import ToolInstance
class AssociateStructurePanel(ToolInstance):
    help = 'help:user/tools/mutationscores.html#associate'

    def __init__(self, session, tool_name = 'Associate Structures with Mutation Data'):

        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self, close_destroys = True)
        self.tool_window = tw
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(parent, margins = (5,0,0,0))

        # Mutation set menu
        msm = self._create_mutation_set_menu(parent)
        layout.addWidget(msm)

        # Table of associated structures
        self._chain_table = sat = self._create_association_table(parent)
        layout.addWidget(sat)
        
        # Associate, Unassociate, Show Alignment
        bf = self._create_buttons(parent)
        layout.addWidget(bf)
                
        tw.manage(placement='side')

        triggers = session.triggers
        from . import ms_data
        ms_data.create_mutation_set_triggers(triggers)
        triggers.add_handler('mutation set added', self._mutation_set_opened)
        triggers.add_handler('mutation set removed', self._mutation_set_closed)
        triggers.add_handler('mutation set structure association changed', self._structure_associations_changed)
        triggers.add_handler('add models', self._open_models_changed)
        triggers.add_handler('remove models', self._open_models_changed)

    @classmethod
    def get_singleton(cls, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, cls, 'Associate Structures with Mutation Data', create=create)

    def _create_mutation_set_menu(self, parent):
        from chimerax.ui.widgets import EntriesRow
        from .ms_data import mutation_scores_names
        mset_names = mutation_scores_names(self.session) + ('set1', 'set2')
        mr = EntriesRow(parent, 'Mutations', mset_names)
        self._mutation_set_menu = msm = mr.values[0]
        msm.shorten_text('middle', 250)
        msmenu = msm.widget.menu()
        msmenu.aboutToShow.connect(lambda *,menu=msmenu: self._menu_about_to_show(menu))
        msmenu.triggered.connect(self._mutation_set_chosen)
        return mr.frame

    def _create_association_table(self, parent):
        chain_assoc_infos = self._chain_association_info()
        ast = AssociateStructuresTable(chain_assoc_infos, parent = parent)
        return ast

    def _chain_association_info(self):
        mset = self._mutation_set
        if mset is None:
            return []

        assoc_chains = mset.associated_chains() if mset else []
        mm_counts = [(chain, self._mismatch_counts(mset, chain)) for chain in assoc_chains]
        mismatch_counts = {chain:mm[0] for chain, mm in mm_counts}
        match_counts = {chain:mm[1] for chain, mm in mm_counts}

        other_chains = []
        from chimerax.atomic import AtomicStructure
        for s in self.session.models.list(type = AtomicStructure):
            other_chains.extend([chain for chain in s.chains if chain not in assoc_chains])

        all_chains = assoc_chains + other_chains
        chain_infos = [{'chain': chain,
                        'associated': ('yes' if chain in assoc_chains else 'no'),
                        'chain_id': chain.string(style = 'command', include_structure = True),
                        'chain_name': chain.description,
                        'sequence_mismatches': mismatch_counts.get(chain),
                        'sequence_matches': match_counts.get(chain),
                        }
                       for chain in all_chains]
        def chain_info_sort_key(ci):
            return (-(ci['sequence_matches'] or 0), ci['chain_id'])
        chain_infos.sort(key = chain_info_sort_key)
        return chain_infos

    def _mismatch_counts(self, mset, chain):
        gapped_mseq, gapped_chain = mset.gapped_chain_alignment(chain)
        gap_char = '.'
        matches = mismatches = 0
        for cm,cc in zip(gapped_mseq.characters, gapped_chain.characters):
            if cm != gap_char and cm != 'X' and cc != gap_char:
                if cc == cm:
                    matches += 1
                else:
                    mismatches += 1

        return mismatches, matches

    # Not used
    def _mismatch_residue_counts(self, mset, assoc_chains):
        ares, arnums = mset.associated_residues()
        mres = {chain:0 for chain in assoc_chains}
        for r in ares:
            mres[r.chain] += 1
        res_type = mset.residue_number_to_amino_acid()
        mmres = {chain:0 for chain in assoc_chains}
        for r,rnum in zip(ares, arnums):
            if res_type[rnum] != r.one_letter_code:
                mmres[r.chain] += 1
        return mmres, mres
    
    def _update_table(self):
        chain_assoc_info = self._chain_association_info()
        self._chain_table.set_table_rows(chain_assoc_info)

    def _mutation_set_opened(self, trigger_name, trigger_data):
        if self.tool_window.tool_instance is None:
            return 'delete handler'	# GUI panel has been destroyed
        if self._mutation_set is None:
            mset = trigger_data
            self.set_mutation_set(mset)

    def _mutation_set_closed(self, trigger_name, trigger_data):
        if self.tool_window.tool_instance is None:
            return 'delete handler'	# GUI panel has been destroyed
        mset = trigger_data
        if mset.name == self._mutation_set_menu.value:
            from .ms_data import mutation_all_scores
            msets = mutation_all_scores(self.session)
            self._mutation_set_menu.value = msets[0].name if msets else ''
            self._update_table()

    def _structure_associations_changed(self, trigger_name, trigger_data):
        if self.tool_window.tool_instance is None:
            return 'delete handler'	# GUI panel has been destroyed
        mset = trigger_data
        if mset == self._mutation_set:
            self._update_table()

    def _open_models_changed(self, trigger_name, models):
        if self.tool_window.tool_instance is None:
            return 'delete handler'	# GUI panel has been destroyed
        if self._mutation_set is None:
            return
        from chimerax.atomic import AtomicStructure
        structures = [model for model in models if isinstance(model, AtomicStructure)]
        if structures:
            self._update_table()

    def _create_buttons(self, parent):
        buttons = [
            ('Associate', self._associate_structures),
            ('Unassociate', self._unassociate_structures),
            ('Show Alignment', self._show_alignment),
            ('Help', self._show_help),
        ]
        from chimerax.ui.widgets import button_row
        f, buttons = button_row(parent, buttons, spacing = 5, button_list = True)
        return f
    
    def _menu_about_to_show(self, menu):
        menu.clear()
        if menu is self._mutation_set_menu.widget.menu():
            from .ms_data import mutation_scores_names
            for ms_name in mutation_scores_names(self.session):
                menu.addAction(ms_name)

    def _mutation_set_chosen(self):
        self._update_table()

    def set_mutation_set(self, mset):
        self._mutation_set_menu.value = mset.name
        self._update_table()
        
    @property
    def _mutation_set(self):
        mutation_set_name = self._mutation_set_menu.value
        if mutation_set_name:
            from .ms_data import mutation_scores
            mset = mutation_scores(self.session, mutation_set_name, raise_error = False)
        else:
            mset = None
        return mset

    def _associate_structures(self):
        self._change_associations('add')
        
    def _unassociate_structures(self):
        self._change_associations('remove')

    def _change_associations(self, add_or_remove):
        chain_infos = self._chain_table.selected
        if len(chain_infos) == 0:
            if add_or_remove == 'remove' or len(self._chain_table.data) == 1:
                chain_infos = self._chain_table.data
            else:
                button_name = 'Associate' if add_or_remove == 'add' else 'Unassociate'
                msg = f'Select one or more table rows then press the {button_name} button'
                self.session.logger.error(msg)
                return
        assoc_state = 'yes' if add_or_remove == 'remove' else 'no'
        chains = [ci.chain for ci in chain_infos if ci.associated == assoc_state]
        mset = self._mutation_set
        if chains and mset:
            from chimerax.atomic import concise_chain_spec
            cspec = concise_chain_spec(chains)
            from chimerax.core.commands import quote_if_necessary
            mset_opt = 'mutationSet ' + quote_if_necessary(mset.name)
            align_opt = 'align true minimumPercentIdentity 0' if add_or_remove == 'add' else ''
            self._run_command(f'mutationscores structure {add_or_remove} {cspec} {align_opt} {mset_opt}')
            self._update_table()

    def _run_command(self, command):
        from chimerax.core.commands import run
        run(self.session, command)

    def _show_alignment(self):
        chain_infos = self._chain_table.selected
        if len(chain_infos) == 0:
            chain_infos = [ci for ci in self._chain_table.data if ci.associated == 'yes']
            if len(chain_infos) == 0:
                msg = f'Select one or more associated structures in the table then press the Show Alignment button'
                self.session.logger.error(msg)
                return
        chains = [ci.chain for ci in chain_infos if ci.associated == 'yes']
        mset = self._mutation_set
        if chains and mset:
            from chimerax.atomic import Sequence
            for chain in chains:
                gapped_mseq, gapped_chain = mset.gapped_chain_alignment(chain)
                seqs = [gapped_mseq, gapped_chain]
                chain_id = chain.string(style = 'command', include_structure = True)
                name = f'{mset.name} and {chain.description} ({chain_id})'
                with self.session.ui.force_float_tools():
                    alignment = self.session.alignments.new_alignment(seqs, name)
                self._highlight_mismatches(alignment.viewers[0])

    def _highlight_mismatches(self, seq_viewer):
        seq1, seq2 = seq_viewer.alignment.seqs
        gap = '.'
        mismatched_rnums = [rnum for rnum, (aa1, aa2) in enumerate(zip(seq1, seq2))
                            if aa1 != gap and aa2 != gap and aa1 != aa2 and aa1 != 'X']

        rm = seq_viewer.region_manager
        blocks = [[seq2, seq2, rnum, rnum] for rnum in mismatched_rnums]
        rm.new_region(name = 'mismatches', blocks = blocks, fill = 'red')

        missing_rnums = [rnum for rnum, aa in enumerate(seq1) if aa == 'X']
        blocks = [[seq1, seq2, rnum, rnum] for rnum in missing_rnums]
        rm.new_region(name = 'missing', blocks = blocks, fill = 'yellow')
        
        # Hide conservation header
        for header in seq_viewer.alignment.headers:
            header.shown = False

    def _show_help(self):
        self._run_command(f'help {self.help}')

# -----------------------------------------------------------------------------
#
from chimerax.ui.widgets import ItemTable
class AssociateStructuresTable(ItemTable):
    def __init__(self, structure_infos, parent = None):
        ItemTable.__init__(self, parent = parent)

        self.add_column('associated', 'associated')
        self.add_column(f'chain id', 'chain_id', multiline_header = False)
        col_name = self.add_column(f'chain name', 'chain_name', multiline_header = False)
        col_mismatches = self.add_column('sequence mismatches', 'sequence_mismatches', format = '%d')
        self.add_column('sequence matches', 'sequence_matches', format = '%d')

        self.set_table_rows(structure_infos)
        self.launch()

        col_name_index = self.columns.index(col_name)
        chain_name_column_width = 260
        self.setColumnWidth(col_name_index, chain_name_column_width)
        self.setAutoScroll(False)  # Avoid click on column scrolling horizontally
        from Qt.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)  # Don't resize whole panel width

    def set_table_rows(self, structure_infos):
        rows = [AssociateStructuresRow(info) for info in structure_infos]
        self.data = rows
        
# -----------------------------------------------------------------------------
#
class AssociateStructuresRow:
    def __init__(self, structure_info):
        self._structure_info = structure_info
    def __getattr__(self, attribute_name):
        return self._structure_info.get(attribute_name)

# -----------------------------------------------------------------------------
#
def show_associate_structure_panel(session, create = True):
    asp = AssociateStructurePanel.get_singleton(session, create=create)
    if asp:
        asp.display(True)
    return asp
