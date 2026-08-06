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

from chimerax.core.tools import ToolInstance
class MutationScoresList(ToolInstance):
    help = 'help:user/tools/mutationscores.html'

    def __init__(self, session, tool_name = 'Mutation Scores'):
        self._selection_order = {}  # Maps (mset, score_name) to counter for ordering selections
        self._selection_count = 0

        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self, close_destroys = False)
        self.tool_window = tw
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(parent, margins = (5,0,0,0))
        
        from Qt.QtWidgets import QListWidget
        class ScoresList(QListWidget):
            def sizeHint(self):
                from Qt.QtCore import QSize
                return QSize(500,50)
        self._mutation_set_list = lw = ScoresList(parent)
        lw.setMinimumHeight(30)
        lw.setSelectionMode(lw.ExtendedSelection)
        lw.itemClicked.connect(self._list_item_clicked)
        layout.addWidget(lw)
        self._update_list()
        from . import ms_data
        ms_data.create_mutation_set_add_remove_triggers(session.triggers,
                                                        self._mutation_set_added,
                                                        self._mutation_set_removed)

        from chimerax.ui.widgets import EntriesRow
        br = EntriesRow(parent,
                        ('Heatmap', self._show_heatmap),
                        ('Scatterplot', self._show_scatterplot),
                        ('Histogram', self._show_histogram),
                        ('Color structure', self._show_color_structure),
                        ('Alphafold structure', self._fetch_alphafold_structure),
                        spacing = 5)
        layout.addWidget(br.frame)

        br2 = EntriesRow(parent,
                         ('Save .csv', self._save_csv),
                         ('Close data', self._close_data),
                         ('Help', self._show_help),
                         spacing = 5)
        layout.addWidget(br2.frame)
                
        tw.manage(placement="side")
    
    @classmethod
    def get_singleton(cls, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, cls, 'Mutation Scores', create=create)

    def _mutation_set_added(self, trigger_name, mset):
        if self.tool_window.ui_area is None:
            return 'delete handler'
        self._update_list()

    def _mutation_set_removed(self, trigger_name, mset):
        if self.tool_window.ui_area is None:
            return 'delete handler'
        self._update_list()
        
    def _update_list(self):
        mset_list = self._mutation_set_list
        from .ms_data import mutation_all_scores
        msets = mutation_all_scores(self.session)
        mset_list.clear()
        row = 0
        from Qt.QtCore import Qt
        for mset in msets:
            mset_list.addItem(mset.name)
            item = mset_list.item(row)
            row += 1
            item.setData(Qt.UserRole, (mset, None))
            for score_name in mset.score_names():
                mset_list.addItem('\t' + score_name)
                item = mset_list.item(row)
                row += 1
                item.setData(Qt.UserRole, (mset, score_name))

    def _list_item_clicked(self, item):
        # Keep track of the order items were selected in so that
        # order can be used in heatmaps and scatter plots.
        from Qt.QtCore import Qt
        mset, score_name = item.data(Qt.UserRole)
        self._selection_order[(mset, score_name)] = self._selection_count
        self._selection_count += 1
        
    def _selected_mutation_scores(self, default_all = True):
        scores = {}	# Maps mutation set name to list of score names
        update_list = False
        open_msets = self._open_mutation_sets
        sel_items = list(self._mutation_set_list.selectedItems())
        from Qt.QtCore import Qt
        sel_scores = [item.data(Qt.UserRole) for item in sel_items]
        sel_scores.sort(key = lambda s: self._selection_order.get(s,0))
        for mset, score_name in sel_scores:
            if mset not in open_msets:
                update_list = True
            else:
                if mset not in scores:
                    scores[mset] = []
                if score_name is not None:
                    scores[mset].append(score_name)
        if update_list:
            self._update_list()
        if len(scores) == 0 and default_all:
            scores = {mset:[] for mset in open_msets}
        return scores

    def _selected_score_names(self):
        scores = self._selected_mutation_scores()
        mset, score_names = tuple(scores.items())[0] if scores else (None, [])
        return mset, score_names

    def _selected_mutation_sets(self):
        return tuple(self._selected_mutation_scores().keys())

    @property
    def _open_mutation_sets(self):
        from .ms_data import mutation_all_scores
        open_msets = mutation_all_scores(self.session)
        return open_msets

    def _show_heatmap(self):
        mset, score_names = self._selected_score_names()
        if mset is None:
            return
        mset_option = self._mset_option(mset.name)
        from chimerax.core.commands import quote_if_necessary
        scores_option = f'scores {quote_if_necessary(",".join(score_names))}' if score_names else ''
        self._run_command(f'mutationscores heatmap {scores_option} {mset_option}')

    def _mset_option(self, mset_name, include_keyword = True):
        if len(self._open_mutation_sets) > 1:
            from chimerax.core.commands import quote_if_necessary
            name = quote_if_necessary(mset_name)
            mset_option = f'mutationSet {name}' if include_keyword else name
        else:
            mset_option = ''
        return mset_option

    def _show_scatterplot(self):
        mset, score_names = self._selected_score_names()
        if mset is None:
            return
        if len(mset.score_names()) < 2:
            from chimerax.core.errors import UserError
            raise UserError(f'Mutation set {mset.name} does not have 2 scores for a scatter plot')
        x_score_name, y_score_name = score_names[:2] if len(score_names) >= 2 else mset.score_names()[:2]
        mset_option = self._mset_option(mset.name)
        from chimerax.core.commands import quote_if_necessary
        self._run_command(f'mutationscores scatterplot {quote_if_necessary(x_score_name)} {quote_if_necessary(y_score_name)} {mset_option}')
    
    def _show_histogram(self):
        mset, score_names = self._selected_score_names()
        if mset is None:
            return
        score_name = score_names[0] if score_names else mset.score_names()[0]
        mset_option = self._mset_option(mset.name)
        from chimerax.core.commands import quote_if_necessary
        self._run_command(f'mutationscores histogram {quote_if_necessary(score_name)} {mset_option}')
    
    def _show_color_structure(self):
        from .ms_color_history import show_structure_coloring_gui
        coloring_gui = show_structure_coloring_gui(self.session)
        mset, score_names = self._selected_score_names()
        if mset is None:
            return
        coloring_gui.set_mutation_set(mset)
        if score_names:
            coloring_gui.set_coloring_score(score_names[0])
    
    def _fetch_alphafold_structure(self):
        mset, score_names = self._selected_score_names()
        if mset is None:
            return
        mset_option = self._mset_option(mset.name, include_keyword = False)
        self._run_command(f'mutationscores alphafold {mset_option}')

    def _save_csv(self):
        mset_names = []
        score_names = []
        mset_to_score_names = self._selected_mutation_scores()
        if len(mset_to_score_names) == 0:
            from chimerax.core.errors import UserError
            raise UserError('There are not mutation scores to save')

        # Classify unique and non-unique score names for making save command options.
        score_name_unique = {}
        from .ms_data import mutation_all_scores
        all_msets = mutation_all_scores(self.session)
        for mset in all_msets:
            for score_name in mset.score_names():
                score_name_unique[score_name] = (score_name not in score_name_unique)

        # Compute mutation sets and score names to use in save command
        for mset, mset_score_names in mset_to_score_names.items():
            if len(mset_score_names) == 0 or set(mset.score_names()) == set(mset_score_names):
                mset_names.append(mset.name)
            else:
                score_names.extend([score_name if score_name_unique.get(score_name) else f'{mset.name}:{score_name}'
                                    for score_name in mset_score_names])

        # Create save command options specifying mutation sets and score names.
        options = []
        if mset_names:
            from .ms_csv_file import csv_join
            mset_names_csv = csv_join(mset_names)
            options.append(f'mutationSets {mset_names_csv}')
        if score_names:
            from .ms_csv_file import csv_join
            score_names_csv = csv_join(score_names)
            options.append(f'scoreNames {score_names_csv}')

        # Show file browser to specify .csv file path
        mset_paths = [mset.path for mset in mset_to_score_names.keys() if mset.path]
        from os.path import dirname
        suggested_path = dirname(mset_paths[0]) if mset_paths else ''
        from Qt.QtWidgets import QFileDialog
        path, ftype  = QFileDialog.getSaveFileName(self.tool_window.ui_area,
                                                   'Save Mutation Scores .csv',
                                                   suggested_path,
                                                   'Mutation scores (*.csv)')
        if not path:
            return 	# Cancelled
        
        # Run save command to create .csv file.
        self._run_command(f'save {path} {" ".join(options)}')
        
    def _close_data(self):
        msets = self._selected_mutation_sets()
        from .ms_data import mutation_scores_close
        for mset in msets:
            mutation_scores_close(self.session, mset.name)

    def _show_help(self):
        self._run_command('help %s' % self.help)

    def _run_command(self, command):
        from chimerax.core.commands import run
        run(self.session, command)
        
    # ---------------------------------------------------------------------------
    # Session save and restore.
    # Even though this tool has no state we want it displayed if it was shown
    # when the session was saved because it is used to access other mutation tools
    # that don't exist in the menus.
    #
    @property
    def SESSION_SAVE(self):
        return self.tool_window.shown
    def take_snapshot(self, session, flags):
        data = {'version': '1'}
        return data
    @classmethod
    def restore_snapshot(cls, session, data):
        msl = show_mutation_scores_list(session)
        return msl

def show_mutation_scores_list(session):
    msl = MutationScoresList.get_singleton(session, create=True)
    msl.display(True)
    return msl
